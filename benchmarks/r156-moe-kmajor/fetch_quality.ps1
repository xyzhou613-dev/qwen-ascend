$ErrorActionPreference = 'Stop'
$qualityRoot = 'D:\Ascend learning\_work\r156\quality-sources'
New-Item -ItemType Directory -Path $qualityRoot -Force | Out-Null
$headers = @{ 'User-Agent' = 'FNX-quality-research' }
$repos = @(
 @{name='gsm8k';repo='openai/grade-school-math';path='grade_school_math/data/test.jsonl';file='gsm8k-test.jsonl'},
 @{name='mbpp';repo='google-research/google-research';path='mbpp/mbpp.jsonl';file='mbpp.jsonl'},
 @{name='humaneval';repo='openai/human-eval';path='data/HumanEval.jsonl.gz';file='HumanEval.jsonl.gz'},
 @{name='cmmlu';repo='haonan-li/CMMLU';path='';file='cmmlu.tar.gz'}
)
$manifest = $repos | ForEach-Object -Parallel {
 $item=$_; $root=$using:qualityRoot
 $html=(Invoke-WebRequest -Uri ('https://github.com/'+$item.repo) -Headers @{'User-Agent'='FNX-quality-research'} -TimeoutSec 30).Content
 $match=[regex]::Match($html,'"currentOid":"([0-9a-f]{40})"')
 if (-not $match.Success) {throw ('Cannot pin source revision: '+$item.repo)}
 $rev=$match.Groups[1].Value
 $url=if ($item.name -eq 'cmmlu') {'https://codeload.github.com/'+$item.repo+'/tar.gz/'+$rev} else {'https://raw.githubusercontent.com/'+$item.repo+'/'+$rev+'/'+$item.path}
 $target=Join-Path $root $item.file
 Invoke-WebRequest -Uri $url -OutFile $target -TimeoutSec 50
 [pscustomobject]@{name=$item.name;revision=$rev;url=$url;file=$item.file;sha256=(Get-FileHash -LiteralPath $target).Hash.ToLowerInvariant();bytes=(Get-Item -LiteralPath $target).Length}
} -ThrottleLimit 4
$hfBefore=Invoke-RestMethod -Uri 'https://huggingface.co/api/datasets/TIGER-Lab/MMLU-Pro' -TimeoutSec 30
$parts=0..11 | ForEach-Object -Parallel {
 $offset=$_*1000; $root=$using:qualityRoot
 $url='https://datasets-server.huggingface.co/rows?dataset=TIGER-Lab%2FMMLU-Pro&config=default&split=test&offset='+$offset+'&length=100'
 $file='mmlu-pro-'+$offset+'.json';$target=Join-Path $root $file
 Invoke-WebRequest -Uri $url -OutFile $target -TimeoutSec 50
 [pscustomobject]@{name='mmlu-pro';url=$url;file=$file;sha256=(Get-FileHash -LiteralPath $target).Hash.ToLowerInvariant();bytes=(Get-Item -LiteralPath $target).Length}
} -ThrottleLimit 4
$hfAfter=Invoke-RestMethod -Uri 'https://huggingface.co/api/datasets/TIGER-Lab/MMLU-Pro' -TimeoutSec 30
if ($hfBefore.sha -ne $hfAfter.sha) { throw 'MMLU-Pro revision changed during fetch' }
foreach ($part in $parts) {$part | Add-Member -NotePropertyName revision -NotePropertyValue $hfBefore.sha}
@{created_utc=[DateTime]::UtcNow.ToString('o');sources=@($manifest)+@($parts);scope='Frozen evaluation source snapshot; not calibration or development tuning'} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $qualityRoot 'source-manifest.json') -Encoding utf8NoBOM
Write-Output ('QUALITY_SOURCES_FETCHED '+(@($manifest).Count+@($parts).Count)+' files')
