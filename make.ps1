#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Windows shim for the Makefile.

.DESCRIPTION
    GNU make does not ship with Windows. This script mirrors every target in the
    Makefile so that development on Windows needs no extra installation, while CI on
    ubuntu runs the Makefile itself.

    The two files must expose exactly the same target names.
    tests/test_tooling.py fails the build if they drift apart.

.EXAMPLE
    .\make.ps1 check
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target = 'help'
)

$ErrorActionPreference = 'Stop'

if ($env:UV) { $uv = $env:UV } else { $uv = 'uv' }

function Invoke-Step {
    param([Parameter(Mandatory)][string[]]$Command)

    Write-Host "==> $($Command -join ' ')" -ForegroundColor Cyan
    if ($Command.Count -gt 1) {
        $rest = $Command[1..($Command.Count - 1)]
        & $Command[0] @rest
    }
    else {
        & $Command[0]
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Stop-NotImplemented {
    param([Parameter(Mandatory)][string]$Name)

    Write-Host "$Name is not implemented until milestone M6. See PROJECT_BRIEF.md." -ForegroundColor Yellow
    exit 1
}

function Invoke-Lint {
    Invoke-Step @($uv, 'run', 'ruff', 'check', '.')
    Invoke-Step @($uv, 'run', 'ruff', 'format', '--check', '.')
}

function Invoke-Typecheck {
    Invoke-Step @($uv, 'run', 'mypy')
}

function Invoke-Test {
    Invoke-Step @($uv, 'run', 'pytest')
}

switch ($Target) {
    'help' {
        Write-Host 'FoldQ targets:'
        Write-Host '  sync         Create or refresh the locked virtual environment'
        Write-Host "  lint         Run ruff's linter and check formatting"
        Write-Host '  format       Rewrite files with the ruff formatter and apply safe fixes'
        Write-Host '  typecheck    Run mypy in strict mode'
        Write-Host '  test         Run the test suite'
        Write-Host '  check        Run everything CI runs (lint, typecheck, test)'
        Write-Host '  clean        Remove caches and build artifacts'
        Write-Host '  bench        (M6) Full benchmark sweep - not implemented yet'
        Write-Host '  bench-fast   (M6) Reduced sweep for CI - not implemented yet'
        Write-Host '  figures      (M6) Regenerate figures from artifacts - not implemented yet'
    }
    'sync' {
        Invoke-Step @($uv, 'sync')
    }
    'lint' {
        Invoke-Lint
    }
    'format' {
        Invoke-Step @($uv, 'run', 'ruff', 'format', '.')
        Invoke-Step @($uv, 'run', 'ruff', 'check', '--fix', '.')
    }
    'typecheck' {
        Invoke-Typecheck
    }
    'test' {
        Invoke-Test
    }
    'check' {
        Invoke-Lint
        Invoke-Typecheck
        Invoke-Test
    }
    'clean' {
        $paths = @(
            '.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov',
            '.coverage', 'coverage.xml', 'build', 'dist'
        )
        foreach ($path in $paths) {
            try { Remove-Item -Recurse -Force -Path $path -ErrorAction Stop } catch {}
        }
        try {
            Get-ChildItem -Recurse -Directory -Filter '__pycache__' |
                Remove-Item -Recurse -Force -ErrorAction Stop
        }
        catch {}
        try {
            Get-ChildItem -Directory -Filter '*.egg-info' |
                Remove-Item -Recurse -Force -ErrorAction Stop
        }
        catch {}
        Write-Host 'Cleaned caches and build artifacts.'
    }
    'bench' {
        Stop-NotImplemented 'make bench'
    }
    'bench-fast' {
        Stop-NotImplemented 'make bench-fast'
    }
    'figures' {
        Stop-NotImplemented 'make figures'
    }
    default {
        Write-Host "Unknown target '$Target'. Run '.\make.ps1 help' for the list." -ForegroundColor Red
        exit 2
    }
}
