# AI Interview Coach

> A production-style AI interview coach built with **prompt engineering**, **chain-of-thought evaluation**, **structured outputs**, **function calling**, **token-aware context management**, and a **two-stage safety layer**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991)](https://openai.com/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-e92063)](https://docs.pydantic.dev/)

---

## Overview

This project implements a fully runnable **AI Interview Coach** that:

- Asks role-specific interview questions (behavioral first, then technical)
- Evaluates each answer using **Chain-of-Thought** reasoning
- Returns **structured JSON scores** (Clarity, Correctness, Completeness) via Pydantic
- Uses **function calling** + a **Tool Arbiter** to invoke scoring tools
- Tracks and trims conversation tokens with **tiktoken**
- Rejects off-topic / jailbreak inputs with a **two-stage safety layer**
- Ends after N questions and computes a session summary

Built as a capstone for Module 1 (Prompt Engineering & LLM Mastery).

---

## Architecture
