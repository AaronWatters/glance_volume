
from __future__ import annotations

import argparse
import neuroglancer
import numpy as np
from .triage import Volume
import H5Gizmos as gz
import socket
from urllib.parse import urlparse, urlunparse, ParseResult

class NeuroglancerComponent:
    def __init__(self, volumes: list[Volume], segmentation: Volume | None = None):
        self.volumes = volumes
        self.segmentation = segmentation
        self.viewer = None

    def neuroglancer_viewer(self) -> neuroglancer.Viewer:
        # configure neuroglancer to listen on all interfaces, so that it can be accessed from a browser on another machine
        neuroglancer.set_server_bind_address("0.0.0.0")
        viewer = neuroglancer.Viewer()
        count = 0
        with viewer.txn() as txn:
            for volume in self.volumes:
                count += 1
                name = "im" + str(count)
                txn.layers[name] = volume.neuroglancer_image_layer()
            if self.segmentation is not None:
                txn.layers["seg"] = self.segmentation.neuroglancer_segmentation_layer()
        self.viewer = viewer
        return viewer
    
    def neuroglancer_url(self) -> str:
        "try to make a URL that will open in a network conneced machine."
        if self.viewer is None:
            self.neuroglancer_viewer()
        local_url = self.viewer.get_viewer_url()
        print(f"Local Neuroglancer URL: {local_url}")
        #result = local_url
        try:
            ip_address = socket.gethostbyname(socket.gethostname())
        except socket.gaierror:
            ip_address = "localhost"
        # use standard url parsing to replace the hostname in the local_url with the actual IP address
        parsed_url = urlparse(local_url)
        port = parsed_url.port
        if port is None:
            raise ValueError(f"Could not parse port number from Neuroglancer URL: {local_url}")
        ip_and_port = f"{ip_address}:{port}"
        print("replacing hostname in URL with IP address:", parsed_url.hostname, "->", ip_address)
        new_url = parsed_url._replace(netloc=ip_and_port) # this deletes the port number, which is needed to access the viewer, so we have to do it manually
        result = urlunparse(new_url)
        print(f"Neuroglancer URL: {result}")
        return result
    
    def neuroglancer_iframe(self, size=1000, height=None) -> str:
        url = self.neuroglancer_url()
        if height is None:
            height = size
        return f'<iframe src="{url}" width="{size}" height="{height}"></iframe>'
    
    def component(self, size=1000, height=None) -> gz.jQueryComponent:
        iframe = self.neuroglancer_iframe(size, height)
        print("neuroglancer iframe:", iframe)
        return gz.Html(iframe)
    
    def run(self):
        component = self.component()
        gz.serve(component.link())


def main(argv: list[str] | None = None, verbose: bool = True) -> int:
    parser = argparse.ArgumentParser(
        prog="volume-glance",
        description="Open one or more image volumes in Neuroglancer.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Image volume path(s).",
    )
    parser.add_argument(
        "-s",
        "--segmentation",
        help="Optional segmentation volume path.",
    )
    args = parser.parse_args(argv)

    volumes = [Volume(path, verbose=verbose) for path in args.paths]
    segmentation = Volume(args.segmentation, verbose=verbose) if args.segmentation else None
    NeuroglancerComponent(volumes, segmentation=segmentation).run()
    return 0
    