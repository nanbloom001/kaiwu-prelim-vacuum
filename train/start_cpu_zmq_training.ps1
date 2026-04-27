$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$envTemplate = Join-Path $PSScriptRoot ".env.cpu-zmq"
$composeTemplate = Join-Path $PSScriptRoot ".docker-compose.cpu-zmq.yaml"
$activeEnv = Join-Path $PSScriptRoot ".env"
$activeCompose = Join-Path $PSScriptRoot ".docker-compose.yaml"

if (-not (Test-Path $envTemplate)) {
    throw "Missing template file: $envTemplate"
}

if (-not (Test-Path $composeTemplate)) {
    throw "Missing template file: $composeTemplate"
}

Copy-Item -LiteralPath $envTemplate -Destination $activeEnv -Force
Copy-Item -LiteralPath $composeTemplate -Destination $activeCompose -Force

Write-Host "[cpu-zmq] restored train/.env and train/.docker-compose.yaml from dedicated templates"
Write-Host "[cpu-zmq] starting distributed training with force recreate"

docker compose `
    --profile distributed `
    --env-file $envTemplate `
    -f $composeTemplate `
    -p kaiwu-train `
    up -d --force-recreate
