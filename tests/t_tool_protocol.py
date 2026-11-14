"""Protocol contract examples; standard library only, no NPU imports."""
import json,os,sys,types,unittest
from pathlib import Path
P=Path(__file__).resolve().parent
runtime=Path(os.environ['FNX_TEST_RUNTIME']) if os.environ.get('FNX_TEST_RUNTIME') else next(root for root in (P/'runtime',P,P.parent) if (root/'fnx').is_dir())
package=types.ModuleType('fnx');package.__path__=[str(runtime/'fnx')];sys.modules['fnx']=package
from fnx.request_validation import RequestError,chat_parameters
from fnx.tool_protocol import select_tools,completion_text,parse_tool_calls,finish_reason
def tool(name='mcp__audit__echo',properties=None,**kw):
    schema=dict(type='object',properties=properties or {},**kw)
    return dict(type='function',function=dict(name=name,parameters=schema))
def xml(name='mcp__audit__echo',**params):
    return '<tool_call>\n<function='+name+'>\n'+''.join('<parameter='+k+'>'+v+'</parameter>\n' for k,v in params.items())+'</function>\n</tool_call>'
def arguments(text,tools):return json.loads(parse_tool_calls(text,tools)[1][0]['function']['arguments'])
class ToolProtocol(unittest.TestCase):
    def body(self,**kw):return dict(messages=[dict(role='user',content='test')],**kw)
    def test_modern_limit_wins(self):self.assertEqual(chat_parameters(self.body(max_tokens=32,max_completion_tokens=512),'model')[1],512)
    def test_null_legacy_does_not_hide_modern(self):self.assertEqual(chat_parameters(self.body(max_tokens=None,max_completion_tokens=512),'model')[1],512)
    def test_null_modern_falls_back(self):self.assertEqual(chat_parameters(self.body(max_tokens=64,max_completion_tokens=None),'model')[1],64)
    def test_default_limit(self):
        for kw in ({},{'max_tokens':None,'max_completion_tokens':None}):self.assertEqual(chat_parameters(self.body(**kw),'model')[1],128)
    def test_invalid_modern_limit(self):
        for value in (0,-1,True,1.5,'32'):
            with self.subTest(value=value),self.assertRaises(RequestError) as raised:chat_parameters(self.body(max_tokens=64,max_completion_tokens=value),'model')
            self.assertEqual(raised.exception.param,'max_completion_tokens')
    def test_auto_and_legacy_default(self):
        t=[tool()]
        self.assertEqual(select_tools(t,'auto'),(t,'',True));self.assertEqual(select_tools(None),( [],'',True))
    def test_none_has_no_definitions_or_parser(self):self.assertEqual(select_tools([tool()],'none'),([], '',False))
    def test_required_keeps_complete_mcp_name(self):self.assertEqual(select_tools([tool()],'required')[1],'<tool_call>\n<function=mcp__audit__echo>\n')
    def test_multiple_required_leaves_name_selection(self):self.assertEqual(select_tools([tool(),tool('second')],'required')[1],'<tool_call>\n')
    def test_specific_filters_tools(self):
        t=tool('second');selected,prefix,parse=select_tools([tool(),t],{'type':'function','function':{'name':'second'}})
        self.assertEqual(selected,[t]);self.assertIn('<function=second>',prefix);self.assertTrue(parse)
    def test_invalid_choices(self):
        for choice in ('invalid',True,[],{}, {'type':'function','function':{'name':'missing'}}):
            with self.subTest(choice=choice),self.assertRaises(RequestError):select_tools([tool()],choice)
    def test_required_without_tools(self):
        with self.assertRaises(RequestError):select_tools([], 'required')
    def test_invalid_declarations(self):
        for tools in ({},[{}],[tool(),tool()],[tool('bad>name')],[dict(type='function',function=dict(name='x',parameters=[]))]):
            with self.subTest(tools=tools),self.assertRaises(RequestError):select_tools(tools)
    def test_structured_values(self):
        t=tool(properties=dict(n={'type':'integer'},f={'type':'number'},b={'type':'boolean'},a={'type':'array'},o={'type':'object'}))
        a=arguments(xml(n='3',f='1.25',b='true',a='[1,"甲"]',o='{"nested":false}'),[t])
        self.assertEqual(a,dict(n=3,f=1.25,b=True,a=[1,'甲'],o={'nested':False}));self.assertIs(type(a['n']),int)
    def test_python_literals(self):
        t=tool(properties=dict(b={'type':'boolean'},v={'type':['integer','null']},o={'type':'object'}))
        self.assertEqual(arguments(xml(b='True',v='None',o="{'ok': False}"),[t]),dict(b=True,v=None,o={'ok':False}))
    def test_string_lexemes_remain_strings(self):
        for raw in ('123','true','null','{"a":1}','[1,2]'):
            with self.subTest(raw=raw):self.assertEqual(arguments(xml(value=raw),[tool(properties={'value':{'type':'string'}})])['value'],raw)
    def test_local_reference(self):
        t=tool(properties={'values':{'$ref':'#/$defs/ints'}},**{'$defs':{'ints':{'type':'array','items':{'type':'integer'}}}})
        self.assertEqual(arguments(xml(values='[1,2]'),[t])['values'],[1,2])
    def test_reference_pointer_escaping(self):
        t=tool(properties={'value':{'$ref':'#/$defs/a~1b~0c'}},**{'$defs':{'a/b~c':{'type':'integer'}}})
        self.assertEqual(arguments(xml(value='9'),[t])['value'],9)
    def test_union_types(self):
        for keyword in ('anyOf','oneOf'):
            t=tool(properties={'v':{keyword:[{'type':'integer'},{'type':'string'}]}})
            self.assertEqual(arguments(xml(v='3'),[t])['v'],3);self.assertEqual(arguments(xml(v='hello'),[t])['v'],'hello')
    def test_boolean_is_not_integer(self):
        raw=xml(n='true');self.assertEqual(parse_tool_calls(raw,[tool(properties={'n':{'type':'integer'}})]),(raw,None))
    def test_boolean_is_not_numeric_enum(self):
        raw=xml(n='true');self.assertEqual(parse_tool_calls(raw,[tool(properties={'n':{'enum':[1]}})]),(raw,None))
    def test_boolean_is_not_numeric_const(self):
        raw=xml(n='true');self.assertEqual(parse_tool_calls(raw,[tool(properties={'n':{'const':1}})]),(raw,None))
    def test_false_parameter_schema(self):
        raw=xml(n='3');self.assertEqual(parse_tool_calls(raw,[tool(properties={'n':False})]),(raw,None))
    def test_nonfinite_number_retained_as_text(self):
        for value in ('NaN','Infinity'):
            raw=xml(n=value);self.assertEqual(parse_tool_calls(raw,[tool(properties={'n':{'type':'number'}})]),(raw,None))
    def test_bad_array_is_not_emitted_as_string(self):
        raw=xml(a='not an array');self.assertEqual(parse_tool_calls(raw,[tool(properties={'a':{'type':'array'}})]),(raw,None))
    def test_required_fields_checked(self):
        raw=xml();self.assertEqual(parse_tool_calls(raw,[tool(properties={'value':{'type':'string'}},required=['value'])]),(raw,None))
    def test_unknown_output_function_retained(self):
        raw=xml(name='other');self.assertEqual(parse_tool_calls(raw,[tool()]),(raw,None))
    def test_duplicate_parameters_retained(self):
        raw=xml(x='1').replace('</function>','<parameter=x>2</parameter></function>')
        self.assertEqual(parse_tool_calls(raw,[tool()]),(raw,None))
    def test_incomplete_xml_preserved(self):
        for raw in ('plain text','<tool_call><function=x>incomplete'):
            self.assertEqual(parse_tool_calls(raw,[tool()]),(raw,None))
    def test_legacy_unschematized_values(self):self.assertEqual(arguments(xml(n='3'),None)['n'],'3')
    def test_multiple_calls_and_surrounding_text(self):
        text='before\n'+xml(n='1')+'\nbetween\n'+xml(n='2')+'\nafter'
        clean,calls=parse_tool_calls(text,[tool(properties={'n':{'type':'integer'}})])
        self.assertEqual(clean,'before\n\nbetween\n\nafter');self.assertEqual(len(calls),2);self.assertNotEqual(calls[0]['id'],calls[1]['id'])
    def test_typed_additional_properties(self):
        t=tool(additionalProperties={'type':'integer'});self.assertEqual(arguments(xml(extra='7'),[t]),{'extra':7})
    def test_additional_properties_false(self):
        raw=xml(extra='7');self.assertEqual(parse_tool_calls(raw,[tool(additionalProperties=False)]),(raw,None))
    def test_literal_eval_does_not_execute(self):
        raw=xml(v="__import__('os').getpid()")
        self.assertEqual(parse_tool_calls(raw,[tool(properties={'v':{'type':'object'}})]),(raw,None))
    def test_bad_reference_preserved(self):
        t=tool(properties={'v':{'$ref':'#/$defs/missing'}});raw=xml(v='3');self.assertEqual(parse_tool_calls(raw,[t]),(raw,None))
    def test_malformed_properties_does_not_crash(self):
        t=tool();t['function']['parameters']['properties']=[];raw=xml(v='3');self.assertEqual(parse_tool_calls(raw,[t]),(raw,None))
    def test_prefix_restored(self):
        p='<tool_call>\n<function=mcp__audit__echo>\n';s='<parameter=text>hi</parameter></function></tool_call>'
        self.assertEqual(arguments(completion_text(p,s),[tool(properties={'text':{'type':'string'}})]),{'text':'hi'})
    def test_restarted_call_is_not_renamed(self):
        raw=xml(name='other');self.assertEqual(completion_text('<tool_call>\n<function=forced>\n',raw),raw)
        self.assertEqual(parse_tool_calls(completion_text('<tool_call>\n<function=forced>\n',raw),[tool('forced')]),(raw,None))
    def test_finish_reasons(self):
        self.assertEqual(finish_reason([1],{2}),'length');self.assertEqual(finish_reason([2],{2}),'stop')
        self.assertEqual(finish_reason([1],{2},[{}]),'tool_calls');self.assertEqual(finish_reason([],{2}),'length')
if __name__=='__main__':unittest.main(verbosity=2)
