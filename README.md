# natlas-guard: Safety Middleware, Quantized Runtime, and Evaluation Harness for N-ATLaS

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Model](https://img.shields.io/badge/Model-NCAIR1%2FN--ATLaS--8B-blue)](https://huggingface.co/NCAIR1/N-ATLaS)
[![Evaluation](https://img.shields.io/badge/Clinical_Miss_Rate-6.25%25-brightgreen)](experiments/)
[![Prescription Suppression](https://img.shields.io/badge/Rx_Suppression-100%25-brightgreen)](experiments/)

**Developed by AMGScale AI LTD** for the National AI Innovation Challenge (NAIC 2026)  
*Track:* Innovation & Enterprise — Developer Infrastructure

---

## 1. Problem Statement & Architecture
While NCAIR's `N-ATLaS` 8B model provides foundational linguistic capabilities across Nigerian languages, deploying it in high-stakes domains (such as healthcare triage) introduces severe risks: unconstrained prescription generation, emergency misclassification, and high compute barriers on local commodity hardware.

`natlas-guard` provides a production-grade infrastructure layer that wraps N-ATLaS:
1. **Dual-Representation Normalizer:** Canonizes non-standard orthography and dialectal Hausa phonetics.
2. **Precedence-Enforced Safety Gate:** Evaluates emergency symptoms and prescription requests deterministically (sub-5ms) before model inference.
3. **Quantized CPU Engine:** Converts N-ATLaS to 4-bit `Q4_K_M` GGUF, achieving 8.2 tokens/sec on standard 8-vCPU commodity hardware.
4. **Tri-Partite Clinical Verifier:** Audits model outputs against grounded guidelines before displaying them to users.