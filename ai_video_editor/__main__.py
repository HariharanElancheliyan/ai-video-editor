import asyncio
import json
import typer
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from .agents.video_editor_agent import VideoEditorAgent
from .config.settings import Settings, setup_logging

console = Console()
app = typer.Typer(
    help="AI-powered video editing tool using LLMs (Ollama / OpenAI / Google)",
    invoke_without_command=True,
    no_args_is_help=False,
)


def _print_tool_call(name: str, args: dict, result: dict) -> None:
    args_preview = json.dumps(args, indent=2, default=str)
    if len(args_preview) > 300:
        args_preview = args_preview[:297] + "..."
    ok = result.get("success") if isinstance(result, dict) else None
    status = "[green]SUCCEED[/green]" if ok else "[red]FAILED[/red]" if ok is False else "[yellow]?[/yellow]"
    console.print(f"[cyan]→ Tool[/cyan] [bold]{name}[/bold] {status} [dim]{args_preview}[/dim]")
    if isinstance(result, dict) and result.get("success") is False:
        err = result.get("error") or result.get("stderr") or ""
        if err:
            console.print(f"  [red]{str(err)[:400]}[/red]")


async def _run_loop(settings: Settings, initial_prompt: str | None = None) -> None:
    agent = VideoEditorAgent(settings)
    provider_label = settings.llm_provider.upper()
    console.print(Panel.fit(
        f"[bold green]AI Video Editor[/bold green]  provider=[magenta]{provider_label}[/magenta]  model=[cyan]{settings.default_model}[/cyan]\n"
        "Type your request. Commands: [yellow]/reset[/yellow] clear history, "
        "[yellow]/exit[/yellow] or [yellow]/quit[/yellow] leave.",
        border_style="green",
    ))
    if settings.allow_system_drive_folder_access:
        console.print(
            "[bold red]⚠  CAUTION:[/bold red] [yellow]System drive (C:\\) folder access is ENABLED. "
            "File operations (read, write, delete, move, copy) can target the system drive. "
            "Proceed with care — unintended changes may affect system stability.[/yellow]"
        )

    pending = initial_prompt
    while True:
        if pending is None:
            try:
                user_input = Prompt.ask("[bold blue]You[/bold blue]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye[/dim]")
                return
        else:
            user_input = pending
            pending = None
            console.print(f"[bold blue]you[/bold blue]: {user_input}")

        if not user_input:
            continue
        cmd = user_input.lower()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            console.print("[dim]bye[/dim]")
            return
        if cmd == "/reset":
            agent.reset()
            console.print("[yellow]conversation reset[/yellow]")
            continue

        try:
            reply = await agent.process_prompt(user_input, on_tool_call=_print_tool_call)
        except Exception as e:
            console.print(f"[red]error:[/red] {e}")
            continue

        if reply:
            console.print(Panel(Markdown(reply), title="assistant", border_style="cyan"))
        else:
            console.print(Panel("[dim](no reply)[/dim]", title="assistant", border_style="cyan"))


def _apply_overrides(settings: Settings, provider: str | None, model: str | None) -> None:
    """Apply CLI overrides to settings in-place."""
    if provider:
        settings.llm_provider = provider  # type: ignore[assignment]
    if model:
        if settings.llm_provider == "openai":
            settings.openai_model = model
        elif settings.llm_provider == "google":
            settings.google_model = model
        else:
            settings.ollama_model = model


@app.callback()
def main(
    ctx: typer.Context,
    prompt: str = typer.Option(None, "-p", "--prompt", help="Optional initial prompt; agent then continues interactively."),
    model: str = typer.Option(None, "-m", "--model", help="LLM model to use"),
    provider: str = typer.Option(None, "--provider", help="LLM provider: ollama, openai, or google"),
    log_level: str = typer.Option(None, "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
):
    """Run the agent in an interactive loop (default)."""
    if ctx.invoked_subcommand is not None:
        return
    settings = Settings()
    setup_logging(log_level or settings.log_level)
    _apply_overrides(settings, provider, model)
    asyncio.run(_run_loop(settings, initial_prompt=prompt))


@app.command()
def edit(
    prompt: str = typer.Option(..., "-p", "--prompt", help="Editing prompt (single-shot, then exit)"),
    model: str = typer.Option(None, "-m", "--model", help="LLM model to use"),
    provider: str = typer.Option(None, "--provider", help="LLM provider: ollama, openai, or google"),
):
    """Run a single prompt through the agent loop and exit."""
    settings = Settings()
    _apply_overrides(settings, provider, model)
    agent = VideoEditorAgent(settings)

    async def run():
        reply = await agent.process_prompt(prompt, on_tool_call=_print_tool_call)
        if reply:
            console.print(Panel(Markdown(reply), title="assistant", border_style="cyan"))
        else:
            console.print(Panel("[dim](no reply)[/dim]", title="assistant", border_style="cyan"))

    asyncio.run(run())


@app.command()
def info(
    file: str = typer.Option(..., "-f", "--file", help="Video file path"),
):
    """Get video information"""
    from .utils.ffmpeg_utils import FFmpegUtils

    settings = Settings()
    ffmpeg = FFmpegUtils(settings.ffmpeg_path, settings.ffprobe_path)
    console.print(ffmpeg.probe(file))


@app.command()
def models(
    provider: str = typer.Option(None, "--provider", help="LLM provider: ollama, openai, or google"),
):
    """List available models for the configured provider"""
    from .core import create_llm_client

    settings = Settings()
    if provider:
        settings.llm_provider = provider  # type: ignore[assignment]
    client = create_llm_client(settings)
    for model in client.list_models():
        rprint(f"[green]{model.get('model', model.get('name', 'unknown'))}[/green]")


if __name__ == "__main__":
    app()