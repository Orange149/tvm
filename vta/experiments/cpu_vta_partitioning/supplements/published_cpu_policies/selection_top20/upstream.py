# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

def submodular_pick(scores, knobs, n_pick, knob_weight=1.0):
    """Run greedy optimization to pick points with regard to both score and diversity.
    DiversityScore = knob_weight * number of unique knobs in the selected set
    Obj = sum(scores[i] for i in pick) + DiversityScore
    Note that this objective function is a monotone submodular function.

    Parameters
    ----------
    scores: Array of float
        score of every points
    knobs: Array of Array of int
        feature vector (tunable knobs) of every points
    n_pick: int
        number of points to pick
    knob_weight: float
        weight of an unique knob feature
    """
    n = len(scores)
    assert n == len(knobs)
    n_knobs = len(knobs[0])

    knobs_set = [set() for _ in range(n_knobs)]

    ret = []
    remain = list(range(len(scores)))

    for _ in range(n_pick):
        max_x = -1
        max_delta = -1e9

        for x in remain:
            tmp_delta = scores[x]
            for i in range(n_knobs):
                if knobs[x][i] not in knobs_set[i]:
                    tmp_delta += knob_weight

            if tmp_delta > max_delta:
                max_delta, max_x = tmp_delta, x

        ret.append(max_x)
        remain.remove(max_x)
        for i in range(n_knobs):
            knobs_set[i].add(knobs[max_x][i])

    return ret
