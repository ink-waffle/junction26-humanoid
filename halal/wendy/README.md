# halal/wendy: the published policy image as a Wendy app

`wendy.json` and a one-line `Dockerfile` wrap the published image, so nothing is rebuilt from source at the venue:
host networking, no GPU, restart policy `no`; the server listens on `$PORT` (8000), which is also the readiness port.

## Publish a tag with both architectures (Mac, once per version)
```bash
bash halal/docker/build.sh v14-arm64 linux/arm64    # this build.sh, in the tree of the handed-in commit; done for v14
R=ghcr.io/ink-waffle/halal-policy
AMD64=$R@sha256:e061f54d6437c359f9b35851f2ae571caf2316e9ceb3b87d12b8229ecfd8b39a    # the hand-in, left as it is
gh auth token | docker login ghcr.io -u ink-waffle --password-stdin
docker tag halal/policy:v14-arm64 $R:v14-arm64 && docker push $R:v14-arm64
docker buildx imagetools create -t $R:v14-multi $AMD64 $R:v14-arm64
docker buildx imagetools inspect $R:v14-multi; docker logout ghcr.io    # must list linux/amd64 and linux/arm64
sed -i '' 's/<tag>/v14-multi/' halal/wendy/Dockerfile
```

## At the venue
```bash
wendy discover                                   # on the robot's hotspot; allow Local Network for the terminal
wendy device info --device <addr>                # architecture, OS, free disk, enrolled or not
wendy device apps list --device <addr>           # other teams' apps and ports
wendy run --prefix halal/wendy --device <addr> --builder docker --detach --wait-ready -y
wendy device logs --app halal-policy --device <addr>
wendy device apps stop halal-policy --device <addr>      # afterwards
wendy device apps remove halal-policy --device <addr>
```

- The runner connects with `--policy ws://<addr>:8000`; for another port change `PORT` and `readiness.tcpSocket.port` in `wendy.json` together.
- `--wait-ready` is documented for WendyOS devices; if the CLI rejects it, drop it and check `apps list` and the logs.
