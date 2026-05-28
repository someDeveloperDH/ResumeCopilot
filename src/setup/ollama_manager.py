# Ollama 자동 설치 / 서버 시작 / 모델 다운로드 관리
# rewriter.py에서 첫 호출 시 자동으로 실행됨
# 이미 설치/실행/다운로드된 경우 해당 단계를 건너뜀

import os
import sys
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request

import requests
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, DownloadColumn, TimeRemainingColumn,
)

console = Console()

OLLAMA_URL  = 'http://localhost:11434'
INSTALL_URL = 'https://ollama.com/install.sh'


# ─────────────────────────────────────────────
# 1단계: Ollama 설치 확인 및 자동 설치
# ─────────────────────────────────────────────

def _is_installed() -> bool:
    return shutil.which('ollama') is not None


def _install() -> None:
    """
    공식 설치 스크립트를 내려받아 실행한다.
    sudo 비밀번호 입력이 필요하므로 반드시 실제 터미널에서 실행해야 한다.
    """
    console.print('[yellow]Ollama가 없습니다. 자동 설치를 시작합니다...[/yellow]')

    # TTY가 없으면 sudo 비밀번호를 입력받을 수 없음
    if not sys.stdin.isatty():
        raise RuntimeError(
            'Ollama 설치는 터미널에서 직접 실행해야 합니다.\n'
            '아래 명령어를 WSL2 터미널에서 직접 입력하세요:\n'
            '  curl -fsSL https://ollama.com/install.sh | sh'
        )

    console.print('[dim]sudo 비밀번호 입력이 필요할 수 있습니다.[/dim]')

    try:
        with urllib.request.urlopen(INSTALL_URL, timeout=30) as resp:
            script = resp.read()
    except Exception as e:
        raise RuntimeError(f'설치 스크립트 다운로드 실패: {e}')

    # input=script 방식은 stdin=PIPE가 되어 sudo가 터미널을 못 잡음
    # → 임시 파일에 저장 후 실행해 stdin을 터미널에 유지
    with tempfile.NamedTemporaryFile(suffix='.sh', delete=False) as f:
        f.write(script)
        tmp_path = f.name

    try:
        os.chmod(tmp_path, 0o755)
        result = subprocess.run(['sh', tmp_path])
    finally:
        os.unlink(tmp_path)

    if result.returncode != 0:
        raise RuntimeError(
            'Ollama 설치 실패.\n'
            '수동 설치: curl -fsSL https://ollama.com/install.sh | sh'
        )

    console.print('[green]✓ Ollama 설치 완료[/green]')


# ─────────────────────────────────────────────
# 2단계: 서버 실행 확인 및 자동 시작
# ─────────────────────────────────────────────

def _is_server_running() -> bool:
    try:
        resp = requests.get(f'{OLLAMA_URL}/api/tags', timeout=2)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _start_server() -> None:
    """
    백그라운드에서 ollama serve를 실행하고 최대 20초 대기한다.
    start_new_session=True로 부모 프로세스 종료 시에도 서버가 유지된다.
    """
    console.print('[yellow]Ollama 서버를 시작합니다...[/yellow]')

    subprocess.Popen(
        ['ollama', 'serve'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    with Progress(SpinnerColumn(), TextColumn('[dim]{task.description}'), console=console) as p:
        task = p.add_task('서버 준비 대기 중...', total=None)
        for _ in range(20):
            time.sleep(1)
            if _is_server_running():
                p.update(task, description='서버 준비 완료')
                break
        else:
            raise RuntimeError(
                'Ollama 서버 시작 실패 (20초 초과).\n'
                '수동 실행: ollama serve'
            )

    console.print('[green]✓ Ollama 서버 실행 중[/green]')


# ─────────────────────────────────────────────
# 3단계: 모델 다운로드 확인 및 자동 pull
# ─────────────────────────────────────────────

def _is_model_available(model: str) -> bool:
    try:
        resp = requests.get(f'{OLLAMA_URL}/api/tags', timeout=3)
        names = [m['name'] for m in resp.json().get('models', [])]
        # 'exaone3.5:7.8b' 또는 'exaone3.5' 형태로 비교
        base = model.split(':')[0]
        return any(base in n for n in names)
    except requests.RequestException:
        return False


def _pull_model(model: str) -> None:
    """
    Ollama 스트리밍 API로 모델을 다운로드하며 Rich 진행바를 표시한다.
    응답 형식: {"status": "downloading", "total": N, "completed": M}
    """
    console.print(f'[yellow]모델 다운로드 중: {model}[/yellow]')

    with requests.post(
        f'{OLLAMA_URL}/api/pull',
        json={'name': model, 'stream': True},
        stream=True,
        timeout=600,
    ) as resp:
        resp.raise_for_status()

        with Progress(
            SpinnerColumn(),
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:

            task = progress.add_task(f'[cyan]{model}', total=None)

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                status = data.get('status', '')

                if 'total' in data and 'completed' in data:
                    # 다운로드 진행률 갱신
                    progress.update(
                        task,
                        total=data['total'],
                        completed=data['completed'],
                        description=f'[cyan]{status}',
                    )
                elif status == 'success':
                    progress.update(task, completed=progress.tasks[task].total or 1,
                                    description='[green]완료')

    console.print(f'[green]✓ 모델 준비 완료: {model}[/green]')


# ─────────────────────────────────────────────
# 통합 진입점
# ─────────────────────────────────────────────

def ensure(model: str) -> None:
    """
    Ollama 사용에 필요한 모든 단계를 순서대로 자동 처리한다.
    각 단계는 이미 완료된 경우 건너뛴다.

        1. Ollama 바이너리 설치 확인
        2. Ollama 서버 실행 확인
        3. 지정 모델 다운로드 확인
    """
    if not _is_installed():
        _install()
    else:
        console.print('[dim]✓ Ollama 설치됨[/dim]')

    if not _is_server_running():
        _start_server()
    else:
        console.print('[dim]✓ Ollama 서버 실행 중[/dim]')

    if not _is_model_available(model):
        _pull_model(model)
    else:
        console.print(f'[dim]✓ 모델 준비됨: {model}[/dim]')
