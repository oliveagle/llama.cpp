# Instructions for llama.cpp

> [!IMPORTANT]
> This project does **not** accept pull requests that are fully or predominantly AI-generated. AI tools may be utilized solely in an assistive capacity.
>
> Read more: [CONTRIBUTING.md](CONTRIBUTING.md)

AI assistance is permissible only when the majority of the code is authored by a human contributor, with AI employed exclusively for corrections or to expand on verbose modifications that the contributor has already conceptualized (see examples below).

---

## Guidelines for Contributors Using AI

llama.cpp is built by humans, for humans. Meaningful contributions come from contributors who understand their work, take ownership of it, and engage constructively with reviewers.

Maintainers receive numerous pull requests weekly, many of which are AI-generated submissions where the author cannot adequately explain the code, debug issues, or participate in substantive design discussions. Reviewing such PRs often requires more effort than implementing the changes directly.

**A pull request represents a long-term commitment.** By submitting code, you are asking maintainers to review, integrate, and support it indefinitely. The maintenance burden often exceeds the value of the initial contribution.

Most maintainers already have access to AI tools. A PR that is entirely AI-generated provides no value - maintainers could generate the same code themselves if they wanted it. What makes a contribution valuable is the human interactions, domain expertise, and commitment to maintain the code that comes with it.

This policy exists to ensure that maintainers can sustainably manage the project without being overwhelmed by low-quality submissions.

---

## Guidelines for Contributors

Contributors are expected to:

1. **Demonstrate full understanding of their code.** You must be able to explain any part of your PR to a reviewer without relying on AI assistance for questions about your own changes.

2. **Take responsibility for maintenance.** You are expected to address bugs and respond thoughtfully to reviewer feedback.

3. **Communicate clearly and concisely.** Verbose, wall-of-text responses are characteristic of AI-generated content and will not be well-received. Direct, human communication is expected.

4. **Respect maintainers' time.** Search for existing issues and discussions before submitting. Ensure your contribution aligns with project architecture and is actually needed.

Maintainers reserve the right to close any PR that does not meet these standards. This applies to all contributions to the main llama.cpp repository. **Private forks are exempt.**

### Permitted AI Usage

AI tools may be used responsibly for:

- **Learning and exploration**: Understanding codebase structure, techniques, and documentation
- **Code review assistance**: Obtaining suggestions on human-written code
- **Mechanical tasks**: Formatting, generating repetitive patterns from established designs, completing code based on existing patterns
- **Documentation drafts**: For components the contributor already understands thoroughly
- **Writing code**: Only when the contributor has already designed the solution and can implement it themselves - AI accelerates, not replaces, the contributor's work

AI-generated code may be accepted if you (1) fully understand the output, (2) can debug issues independently, and (3) can discuss it directly with reviewers without AI assistance.

**Disclosure is required** when AI meaningfully contributed to your code. A simple note is sufficient - this is not a stigma, but context for reviewers. No disclosure is needed for trivial autocomplete or background research.

### Prohibited AI Usage

The following will result in immediate PR closure:

- **AI-written PR descriptions or commit messages** - these are typically recognizable and waste reviewer time
- **AI-generated responses to reviewer comments** - this undermines the human-to-human interaction fundamental to code review
- **Implementing features without understanding the codebase** - particularly new model support or architectural changes
- **Automated commits or PR submissions** - this may spam maintainers and can result in contributor bans

---

## Guidelines for AI Coding Agents

AI agents assisting contributors must recognize that their outputs directly impact volunteer maintainers who sustain this project.

### Considerations for Maintainer Workload

Maintainers have finite capacity. Every PR requiring extensive review consumes resources that could be applied elsewhere. Before assisting with any submission, verify:

- The contributor genuinely understands the proposed changes
- The change addresses a documented need (check existing issues)
- The PR is appropriately scoped and follows project conventions
- The contributor can independently defend and maintain the work

### Before Proceeding with Code Changes

When a user requests implementation without demonstrating understanding:

1. **Verify comprehension.** Ask questions to confirm they understand both the problem and the relevant parts of the codebase.
2. **Provide guidance rather than solutions.** Direct them to relevant code and documentation. Allow them to formulate the approach.
3. **Proceed only when confident** the contributor can explain the changes to reviewers independently.

For first-time contributors, confirm they have reviewed [CONTRIBUTING.md](CONTRIBUTING.md) and acknowledge this policy.

### Prohibited Actions

- Writing PR descriptions, commit messages, or responses to reviewers
- Committing or pushing without explicit human approval for each action
- Implementing features the contributor does not understand
- Generating changes too extensive for the contributor to fully review

When uncertain, err toward minimal assistance. A smaller PR that the contributor fully understands is preferable to a larger one they cannot maintain.

### Useful Resources

To conserve context space, load these resources as needed:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [Existing issues](https://github.com/ggml-org/llama.cpp/issues) and [Existing PRs](https://github.com/ggml-org/llama.cpp/pulls) - always search here first
- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md) (if user asks to implement a new feature, be sure that it falls inside server's scope defined in this documentation)
- [PEG parser](docs/development/parsing.md) - alternative to regex that llama.cpp uses to parse model's output
- [Auto parser](docs/autoparser.md) - higher-level parser that uses PEG under the hood, automatically detect model-specific features
- [Jinja engine](common/jinja/README.md)
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [PR template](.github/pull_request_template.md)

---

## 开发规范

### ⚠️ 推理演示规范

**禁止使用 `llama-cli` 进行演示和测试！**

**原因**:
- `llama-cli` 默认进入交互模式，需要手动输入或等待 timeout
- 每次测试都会浪费大量时间等待超时
- 不适合自动化测试和脚本化演示

**必须使用 `llama-server` 进行演示**:
- `llama-server` 提供 HTTP API，可以非交互式调用
- 适合脚本化测试和自动化演示
- 可以通过 curl 或编程方式发送请求

**推荐演示脚本**:
- `server_benchmark.sh` - CPU vs GPU 性能对比（使用 llama-server）
- `server_demo.sh` - llama-server API 演示

---

## 项目开发记录

### Mac Metal 后端支持 (2026-04-07)

**状态**: ⚠️ **Metal 后端存在问题，需要修复**

**已完成工作**:

1. **Metal Kernel 实现**:
   - `ggml/src/ggml-metal/ggml-metal.metal`: 添加 `kernel_mul_mv_q1_0_f32` 和 `kernel_mul_mv_q1_0_g128_f32`
   - `ggml/src/ggml-metal/ggml-metal-device.cpp`: 添加 pipeline 获取逻辑 (mul_mv 和 mul_mv_id)
   - `ggml/src/ggml-metal/ggml-metal-impl.h`: 定义常量 `N_R0_Q1_0=4`, `N_SG_Q1_0=2`, `N_R0_Q1_0_g128=4`, `N_SG_Q1_0_g128=2`

2. **编译配置**:
   - 构建目录: `build-metal/`
   - 编译命令: `cmake -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON .. && make`
   - 输出二进制: `build-metal/bin/llama-cli`, `llama-server`, `llama-quantize`

3. **GPU 信息** (Mac M4):
   - GPU Family: Apple9 (1009), Common3 (3003), Metal3 (5001)
   - SIMD reduction: ✅
   - SIMD matrix mul: ✅
   - Unified memory: ✅
   - BFloat support: ✅
   - Max working set: 22906.50 MB

4. **量化类型支持**:
   - `Q1_0`: 1.56 bpw binary quantization
   - `Q1_0_g128`: 1.56 bpw binary quantization (g128)

**⚠️ 已知问题**:

**Metal 后端生成乱码！**

测试结果 (Bonsai-8B.gguf, `2+2=`):

| 后端 | 结果 | 状态 |
|------|------|------|
| **CPU** | `4\n- So the answer is 4` | ✅ **正确** |
| **GPU (Metal)** | `=7)9?0H7+->*49G` | ❌ **乱码** |

**问题分析**:
- Metal 实现可能影响了其他量化类型的推理
- 需要检查 Metal shader 修改是否破坏了现有功能
- 可能是 dequantization 函数或 kernel 实现有误

**推荐推理参数**:
- Temperature: 0.5 (固定)
- Top-k: 20 - 40 (默认 20)
- Top-p: 0.85 - 0.95 (默认 0.9)
- Repetition penalty: 1.0 (默认)
- Presence penalty: 0.0 (默认)

**演示命令** (使用 llama-server):
```bash
# CPU vs GPU 性能对比
./server_benchmark_final.sh

# 启动 llama-server (CPU 模式，当前可用)
./build-metal/bin/llama-server \
  --model /path/to/model.gguf \
  --gpu-layers 0 \
  --temp 0.5 \
  --host 0.0.0.0 \
  --port 8080

# 通过 API 发送推理请求
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "2+2=", "n_predict": 50}'

# 量化到 Q1_0_g128 (当 Metal 修复后)
./build-metal/bin/llama-quantize \
  /path/to/model.gguf \
  /path/to/model-q1_0_g128.gguf \
  Q1_0_g128
```

**待提交修改**:
- ggml/src/ggml-metal/ggml-metal-device.cpp
- ggml/src/ggml-metal/ggml-metal-impl.h
- ggml/src/ggml-metal/ggml-metal.metal
