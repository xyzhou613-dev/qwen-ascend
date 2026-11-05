"""CPU-only HTTP tests with the real tokenizer and a deterministic fake engine."""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest

os.environ['FNX_PIPELINE']='0'
os.environ['FNX_CTX']='262144'
os.environ['FNX_ENABLE_THINKING']='0'
from fastapi.testclient import TestClient
from transformers import AutoTokenizer
import torch
from fnx import server as S

tok=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next',local_files_only=True)


class FakeModel:
    def __init__(self):self.logits=torch.zeros(1,len(tok));self.transport_stats={}
    def reset(self):pass
    def step_pad(self,*args,**kw):time.sleep(.002)
    def row_logits(self):return self.logits


class FakeEngine:
    def __init__(self):
        self.tok=tok;self.m=FakeModel();self.conv_slots=[];self.n_req=0
        self.stop_ids={tok.eos_token_id};self.last_prof={};self.text='𠮷🧪🚀 café Ελληνικά';self.active=0
    def generate(self,ids,max_new,*args):
        assert self.active==0,'generation overlap'
        self.active+=1
        try:
            for t in (tok.encode(self.text,add_special_tokens=False)+[tok.eos_token_id])[:max_new]:
                time.sleep(.002);yield ('t',t)
            self.last_prof={'n_prefill':len(ids)-1,'prefill_ms':1,'prefill_tok_s':1000,'tpot_ms':2,'ttft_ms':3}
        finally:self.active-=1


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine=FakeEngine();S.STATE['e']=cls.engine
        cls.context=TestClient(S.app);cls.client=cls.context.__enter__()
    @classmethod
    def tearDownClass(cls):cls.context.__exit__(None,None,None)
    def setUp(self):self.engine.text='𠮷🧪🚀 café Ελληνικά'
    def body(self,**kw):
        return dict(messages=[{'role':'user','content':'请照抄测试文本'}],max_tokens=128,temperature=0,**kw)
    def streamed(self,body):
        r=self.client.post('/v1/chat/completions',json=dict(body,stream=True));self.assertEqual(r.status_code,200)
        content=[];reason=None;done=False
        for line in r.text.splitlines():
            if not line.startswith('data: '):continue
            if line[6:]=='[DONE]':done=True;continue
            d=json.loads(line[6:]);c=d['choices'][0];content.append(c['delta'].get('content',''))
            if c['finish_reason'] is not None:reason=c['finish_reason']
        return ''.join(content),reason,done
    def test_engine_lookup_does_not_acquire_generation_lock(self):
        with S.LOCK:self.assertIs(S.get_engine(),self.engine)
    def test_unicode_stream_matches_nonstream(self):
        r=self.client.post('/v1/chat/completions',json=self.body()).json()
        text,reason,done=self.streamed(self.body())
        self.assertEqual(text,r['choices'][0]['message']['content']);self.assertNotIn('\ufffd',text)
        self.assertEqual(reason,'stop');self.assertTrue(done)
    def test_unicode_every_truncation_prefix(self):
        ids=tok.encode(self.engine.text,add_special_tokens=False)
        for cap in range(1,len(ids)+1):
            body=dict(self.body(),max_tokens=cap)
            plain=self.client.post('/v1/chat/completions',json=body).json()
            text,reason,done=self.streamed(body)
            self.assertEqual(text,plain['choices'][0]['message']['content'],cap)
            self.assertEqual(reason,'length');self.assertTrue(done)
    def test_english_and_json(self):
        for text in ('Alpha beta 0123456789','{"status":"成功","emoji":"🧪"}'):
            self.engine.text=text;got,reason,done=self.streamed(self.body());self.assertEqual(got,text)
    def test_parallel_streams_and_health(self):
        with cf.ThreadPoolExecutor(max_workers=3) as ex:
            futures=[ex.submit(self.streamed,self.body()) for _ in range(3)]
            for _ in range(5):
                self.assertEqual(self.client.get('/health').status_code,200);time.sleep(.005)
            for f in futures:self.assertEqual(f.result(timeout=10)[0],self.engine.text)
    def test_more_streams_than_default_worker_pool(self):
        self.engine.text='OK'
        with cf.ThreadPoolExecutor(max_workers=45) as ex:
            futures=[ex.submit(self.streamed,self.body()) for _ in range(45)]
            for _ in range(8):
                self.assertEqual(self.client.get('/health').status_code,200);time.sleep(.005)
            for f in futures:self.assertEqual(f.result(timeout=15)[0],'OK')
    def test_bad_json(self):self.assertEqual(self.client.post('/v1/chat/completions',content='{').status_code,400)
    def test_nonobject_body(self):self.assertEqual(self.client.post('/v1/chat/completions',json=[]).status_code,400)
    def test_missing_messages(self):self.assertEqual(self.client.post('/v1/chat/completions',json={}).status_code,400)
    def test_empty_messages(self):self.assertEqual(self.client.post('/v1/chat/completions',json={'messages':[]}).status_code,400)
    def test_invalid_message_content(self):
        r=self.client.post('/v1/chat/completions',json={'messages':[{'role':'user','content':None}]});self.assertEqual(r.status_code,400)
    def test_nontext_content_part(self):
        b=self.body();b['messages'][0]['content']=[{'type':'image_url','image_url':'x'}]
        self.assertEqual(self.client.post('/v1/chat/completions',json=b).status_code,400)
    def test_text_content_parts(self):
        b=self.body();b['messages'][0]['content']=[{'type':'text','text':'hello'}]
        self.assertEqual(self.client.post('/v1/chat/completions',json=b).status_code,200)
    def test_wrong_model(self):self.assertEqual(self.client.post('/v1/chat/completions',json=self.body(model='wrong')).status_code,404)
    def test_n_unsupported(self):self.assertEqual(self.client.post('/v1/chat/completions',json=self.body(n=2)).status_code,400)
    def test_stream_type(self):
        b=dict(self.body(),stream='false');self.assertEqual(self.client.post('/v1/chat/completions',json=b).status_code,400)
    def test_max_tokens_validation(self):
        for value in (0,-1,True,'4',1.2):
            self.assertEqual(self.client.post('/v1/chat/completions',json=dict(self.body(),max_tokens=value)).status_code,400)
    def test_temperature_validation(self):
        for value in (-1,3,'bad',True):
            self.assertEqual(self.client.post('/v1/chat/completions',json=dict(self.body(),temperature=value)).status_code,400)
    def test_nan_temperature(self):
        b=self.body();b['temperature']=float('nan')
        self.assertEqual(self.client.post('/v1/chat/completions',content=json.dumps(b)).status_code,400)
    def test_budget_overflow(self):self.assertEqual(self.client.post('/v1/chat/completions',json=dict(self.body(),max_tokens=262145)).status_code,400)
    def test_completion_token_alias(self):
        b=self.body();del b['max_tokens'];b['max_completion_tokens']=1
        self.assertEqual(self.client.post('/v1/chat/completions',json=b).json()['usage']['completion_tokens'],1)
    def test_loss_bad_cap(self):self.assertEqual(self.client.post('/fnx/loss',json={'text':'hello','max_tokens':0}).status_code,400)
    def test_loss_bad_body(self):self.assertEqual(self.client.post('/fnx/loss',json=[]).status_code,400)
    def test_loss_and_stream(self):
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            f=ex.submit(self.streamed,self.body())
            loss=ex.submit(self.client.post,'/fnx/loss',json={'text':'hello world again','max_tokens':2})
            self.assertEqual(f.result(timeout=10)[0],self.engine.text)
            self.assertEqual(loss.result(timeout=10).status_code,200)
    def test_selfcheck_bad_cap(self):self.assertEqual(self.client.post('/fnx/selfcheck',json={'n':0}).status_code,400)


if __name__=='__main__':unittest.main(verbosity=2)
