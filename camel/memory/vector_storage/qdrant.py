# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from dataclasses import asdict
from typing import List, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    CollectionStatus,
    Distance,
    PointIdsList,
    PointStruct,
    UpdateStatus,
    VectorParams,
)

import camel.memory.vector_storage.base as base


class Qdrant():

    def __init__(
        self,
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        if url is not None:
            self.client = QdrantClient(url=url, api_key=api_key)
        elif path is not None:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")

    def create_collection(
        self,
        collection: str,
        size: int,
        distance: base.Distance = base.Distance.DOT,
        **kwargs,
    ) -> None:
        distance_map = {
            base.Distance.DOT: Distance.DOT,
            base.Distance.COSINE: Distance.COSINE,
            base.Distance.EUCLIDEAN: Distance.EUCLID,
        }
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=size,
                                        distance=distance_map[distance]),
            **kwargs,
        )

    def delete_collection(
        self,
        collection: str,
        **kwargs,
    ) -> None:
        self.client.delete_collection(collection_name=collection, **kwargs)

    def check_collection(self, collection: str) -> None:
        # TODO: check more information
        collection_info = self.client.get_collection(
            collection_name=collection)
        if collection_info.status != CollectionStatus.GREEN:
            raise RuntimeWarning(f"Qdrant collection \"{collection}\" status: "
                                 f"{collection_info.status}")

    def add_vectors(
        self,
        collection: str,
        vectors: List[base.VectorRecord],
    ) -> None:
        qdrant_points = [
            PointStruct(
                id=p.id if p.id is not None else str(uuid4()),
                **asdict(p),
            ) for p in vectors
        ]
        op_info = self.client.upsert(
            collection_name=collection,
            points=qdrant_points,
            wait=True,
        )
        if op_info.status != UpdateStatus.COMPLETED:
            raise RuntimeError(
                "Failed to add vectors in Qdrant, operation info: "
                f"{op_info}")

    def delete_vectors(self, collection: str,
                       ids: List[base.VectorRecord]) -> None:
        op_info = self.client.delete(
            collection_name=collection,
            points_selector=PointIdsList([p.id for p in ids]),
            wait=True,
        )
        if op_info.status != UpdateStatus.COMPLETED:
            raise RuntimeError(
                "Failed to delete vectors in Qdrant, operation info: "
                f"{op_info}")

    def search(
        self,
        query_vector: base.VectorRecord,
        limit: int = 3,
    ) -> List[base.VectorRecord]:
        # TODO: filter
        if query_vector.vector is None:
            raise RuntimeError("Searching vector cannot be None")
        search_result = self.client.search(
            collection_name="test_collection",
            query_vector=query_vector.vector,
            with_payload=True,
            limit=limit,
        )
        # TODO: including score?
        result_records = []
        for res in search_result:
            result_records.append(
                base.VectorRecord(
                    id=res.id,
                    payload=res.payload,
                ))

        return result_records