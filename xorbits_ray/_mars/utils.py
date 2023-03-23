# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from urllib.parse import urlparse


def is_on_ray(ctx):
    from .services.task.execution.ray.context import (
        RayExecutionContext,
        RayExecutionWorkerContext,
    )

    # There are three conditions
    #   a. mars backend
    #   b. ray backend(oscar), c. ray backend(dag)
    # When a. or b. is selected, ctx is an instance of ThreadedServiceContext.
    #   The main difference between them is whether worker address matches ray scheme.
    #   To avoid duplicated checks, here we choose the first worker address.
    # When c. is selected, ctx is an instance of RayExecutionContext or RayExecutionWorkerContext,
    #   while get_worker_addresses method isn't currently implemented in RayExecutionWorkerContext.
    try:
        worker_addresses = ctx.get_worker_addresses()
    except AttributeError:  # pragma: no cover
        assert isinstance(ctx, RayExecutionWorkerContext)
        return True
    return isinstance(ctx, RayExecutionContext) or is_ray_address(worker_addresses[0])


def is_ray_address(address: str) -> bool:
    if urlparse(address).scheme == "ray":
        return True
    else:
        return False


def register_ray_serializer(obj_type, serializer=None, deserializer=None):
    try:
        import ray

        try:
            ray.register_custom_serializer(
                obj_type, serializer=serializer, deserializer=deserializer
            )
        except AttributeError:  # ray >= 1.0
            try:
                from ray.worker import global_worker

                global_worker.check_connected()
                context = global_worker.get_serialization_context()
                context.register_custom_serializer(
                    obj_type, serializer=serializer, deserializer=deserializer
                )
            except AttributeError:  # ray >= 1.2.0
                ray.util.register_serializer(
                    obj_type, serializer=serializer, deserializer=deserializer
                )
    except ImportError:
        pass
