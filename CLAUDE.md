# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI/ML notebook project focused on batch Excel data processing and model training workflows. The project is in early development stage with modular Python architecture designed for:
- Batch Excel reading and preprocessing
- Multi-step time series data processing
- XGBoost/LightGBM model training
- Wide table data transformations

## Architecture

The project follows a modular Python package structure:

```
ai-notebook/
├── src/               # Core source code module，trench宽和深的预测模型的训练、验证、测试代码
├── data_insight/      # Data analysis and insights generation，看单个wafer各个步骤的统计学特征及不同wafer同一步骤的统计学特征
├── db/                # Database operations and connectors
├── utils/             # Utility functions and helpers
├── wide_table/        # Wide table data processing，将wafer相关数据处理成大宽表
├── test/              # Test suite
└── snippets/          # Code snippets and reference materials
```

Key architectural patterns:
- Each module contains `__init__.py` for package initialization
- Domain-driven module organization
- Standard Python package structure for maintainability

## Development Commands

This project currently lacks configuration files. As it develops, expect to add:
- `requirements.txt` or `pyproject.toml` for dependencies
- Test framework configuration
- Linting and formatting tools (black, flake8, mypy)
- Build and deployment scripts

## Reference Materials

The `snippets/qa.md` file contains valuable guidance on:
- Batch Excel processing workflows
- Multi-step time series prediction patterns
- XGBoost/LightGBM integration approaches
- ML tools comparison (PyCaret, Qlib, H2O AutoML)

When implementing features, consult this file for proven patterns and tool recommendations.

## Current State

This is a greenfield project with module structure defined but implementation pending. When adding functionality:
- Follow the established module boundaries
- Use pandas for Excel processing
- Consider the time series and wide table use cases documented in snippets
- Implement proper error handling for batch processing operations