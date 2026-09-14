# Root review

This run is retained only as an invalidated audit trail. It used the local-only
`rpc.LocalSession()` API. No network, SSH, RPC server, or board was contacted,
but the call still violates the strict P4j no-RPC protocol. Downstream work must
consume run02 only.
