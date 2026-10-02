# TROPT (vendored, early version)

This directory vendors an **early version of TROPT (v0.0.1a1)**, the discrete-text
optimizer used by TellTail-OPT (`opt/`, `demo optimize`). It is included **with the
authors' permission**.

> ⚠️ Use *this* vendored copy — do **not** `pip install` TROPT from upstream. TellTail-OPT
> targets this early version's API; the current upstream release differs. It is deliberately
> **not** in `requirements.txt`; install it explicitly:
>
> ```bash
> pip install -e third_party/tropt
> ```

Project: https://github.com/matanbt/TROPT

If you use TROPT, please cite:

```bibtex
@misc{bentov2026troptopenframeworkunifying,
      title={TROPT: An Open Framework for Unifying and Advancing Discrete Text Optimization},
      author={Matan Ben-Tov and Mahmood Sharif},
      year={2026},
      eprint={2606.23496},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.23496},
}
```
