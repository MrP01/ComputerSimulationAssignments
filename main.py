#!/usr/bin/env python3
"""Entrypoint for running exercises"""
import importlib

import typer


def main(task: str):
    """Run the given task"""
    typer.echo(f"Running task {task}.")
    module = importlib.import_module(task)
    module.main()


if __name__ == "__main__":
    typer.run(main)
