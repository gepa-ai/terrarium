# Terrarium

**Terrarium** is a benchmarking framework for evaluating AI-driven program synthesis and evolution systems.

It provides a standard way to define benchmark tasks and run any evolution system against them with controlled, measured evaluation budgets.

## Overview

Terrarium is part of the [GEPA](https://github.com/gepa-ai/gepa) project — a framework for Genetic Evolution of Program Abstractions. While GEPA focuses on evolving programs using language models, Terrarium provides the evaluation infrastructure to measure and compare different evolution strategies.

## Links

- 📦 **GEPA Repository**: [https://github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa)
- 🌐 **GEPA Website**: [https://gepa-ai.github.io/gepa/](https://gepa-ai.github.io/gepa/)

## Installation

```bash
pip install terrarium
```

## Features

- Define benchmark tasks with a standard interface
- Run any evolution system through a unified `EvalServer`
- Enforce evaluation budgets server-side
- Support for in-process and external adapters

## License

This project is licensed under the MIT License.