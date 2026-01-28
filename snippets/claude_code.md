

## envs

```Powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:ANTHROPIC_API_KEY = "sk-"
$env:ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
$env:ANTHROPIC_MODEL= "deepseek-reasoner"
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = "deepseek-reasoner"
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = "deepseek-reasoner"
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = "deepseek-reasoner"
$env:CLAUDE_CODE_SUBAGENT_MODEL = "deepseek-reasoner"
$env:CLAUDE_PACKAGE_MANAGER = pnpm
$env:GITHUB_URL = "https://"
$env:SHELL = ""
```

```curl
curl --request POST \
  --url https://api.deepseek.com/anthropic/v1/messages \
  --header 'Authorization: Bearer sk-' \
  --header 'Content-Type: application/json' \
  --data '{
	"model": "deepseek-reasoner",
	"max_tokens": 8192,
	"system": "You are a helpful coding assistant.",
	"messages": [
		{
			"role": "user",
			"content": "你擅长什么"
		}
	],
	"stream": true,
	"temperature": 0.7
}'
```