
from __future__ import annotations

import argparse
import neuroglancer
import numpy as np
from .triage import Volume, neuroglancer_colored_shader
import H5Gizmos as gz
import socket
from urllib.parse import urlparse, urlunparse, ParseResult

class NeuroglancerComponent:
    def __init__(self, volumes: list[Volume], segmentation: Volume | None = None):
        self.volumes = volumes
        self.segmentation = segmentation
        self.viewer = None
        self.name_to_volume = {}

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
                self.name_to_volume[name] = volume
            if self.segmentation is not None:
                txn.layers["seg"] = self.segmentation.neuroglancer_segmentation_layer()
        self.viewer = viewer
        return viewer
    
    def labels_range(self, *ignored):
        if self.segmentation is None:
            return None
        if self.viewer is None:
            self.neuroglancer_viewer()
        with self.viewer.txn() as txn:
            segmentation = self.segmentation
            for volume_name in self.name_to_volume:
                volume = self.name_to_volume[volume_name]
                volume_color = volume.color
                min_label, max_label = volume.sliced_range(segmentation)
                print(f"labels range for volume {volume_name}: {min_label} - {max_label}")
                layer = txn.layers[volume_name]
                if layer is not None:
                    layer.shader = neuroglancer_colored_shader(volume_color, min_label, max_label)
        return None
    
    def full_range(self, *ignored):
        if self.viewer is None:
            self.neuroglancer_viewer()
        with self.viewer.txn() as txn:
            for volume_name in self.name_to_volume:
                volume = self.name_to_volume[volume_name]
                volume_color = volume.color
                min_value, max_value = volume.full_range()
                print(f"full range for volume {volume_name}: {min_value} - {max_value}")
                layer = txn.layers[volume_name]
                if layer is not None:
                    layer.shader = neuroglancer_colored_shader(volume_color, min_value, max_value)
        return None
    
    def select_all_labels(self, *ignored):
        if self.segmentation is None:
            return # nothing to select
        if self.viewer is None:
            self.neuroglancer_viewer()
        with self.viewer.txn() as txn:
            seg = txn.layers["seg"]
            if seg is not None:
                labels = self.segmentation.labels()
                print(f"selecting all {len(labels)} labels in segmentation layer")
                seg.visible_segments = labels

    def unselect_all_labels(self, *ignored):
        if self.segmentation is None:
            return # nothing to unselect
        if self.viewer is None:
            self.neuroglancer_viewer()
        with self.viewer.txn() as txn:
            seg = txn.layers["seg"]
            if seg is not None:
                print("unselecting all labels in segmentation layer")
                seg.visible_segments = []
    
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
        #self.select_all_labels()
        iframe = gz.Html(iframe)
        select_all_button = gz.Button("Select all labels", on_click=self.select_all_labels)
        unselect_all_button = gz.Button("Unselect all labels", on_click=self.unselect_all_labels)
        full_range_button = gz.Button("Set full range", on_click=self.full_range)
        set_sliced_range_button = gz.Button("Set sliced range", on_click=self.labels_range)
        buttons = [select_all_button, unselect_all_button, full_range_button, set_sliced_range_button]
        result = gz.Stack([
            iframe,
            buttons,
        ])
        def on_start():
            print("neuroglancer component started, selecting all labels")
            self.select_all_labels()
            self.labels_range()
        result.call_when_started(on_start)
        return result
    
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
    