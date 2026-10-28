# WebNN Specification Reference

Download or refresh the live W3C WebNN Bikeshed source from the repository root:

```bash
make fetch-webnn-spec
```

The recipe saves the source to `target/webnn-index.bs`. The download is a local
build artifact, excluded from version control. It comes from the
[WebNN main branch](https://github.com/webmachinelearning/webnn/blob/main/index.bs),
which can change independently of rustnn.

The source is human-readable and contains operation definitions, signatures, and
validation algorithms. To find builder signatures:

```bash
rg 'MLOperand \w+\(' target/webnn-index.bs
```

## Cross-References

- [Generated backend operator support](../development/backend-operator-support.md)
- [Published WebNN specification](https://www.w3.org/TR/webnn/)
- [Specification repository](https://github.com/webmachinelearning/webnn)
