# Copyright 2022 Hiori Kino
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.

class zerofillStr:
    """returns string 023 if i=23 and if the total number is 3 digits.
    """

    def __init__(self, npattern):
        self._set_number_of_zerofill(npattern)

    def str(self, i):
        return str(i).zfill(self._nzerofills)

    def _set_number_of_zerofill(self, npattern):

        nzero = 1

        while True:
            npattern //= 10
            if npattern == 0:
                break
            nzero += 1

        self._nzerofills = nzero


def parse_job_times(text) -> dict:
    """start / end time and thread counts from the stdout of alm / anphon.

    Args:
        text (str or list): the output file content (or its lines).

    Returns:
        dict: job_started, job_finished (ISO 8601, local time), elapsed_seconds, and when printed
        num_openmp_threads, num_mpi_processes. Missing items are left out.
    """
    from datetime import datetime
    lines = text.splitlines() if isinstance(text, str) else text
    result = {}
    for line in lines:
        s = line.strip()
        for key, head in (("job_started", "Job started at"), ("job_finished", "Job finished at")):
            if s.startswith(head):
                try:
                    result[key] = datetime.strptime(s[len(head):].strip(), "%a %b %d %H:%M:%S %Y").isoformat()
                except ValueError:
                    result[key] = s[len(head):].strip()
        if "OpenMP threads" in s:
            result["num_openmp_threads"] = int(s.replace("=", ":").split(":")[-1])
        elif "number of MPI processes" in s:
            result["num_mpi_processes"] = int(s.split(":")[-1])
    if "job_started" in result and "job_finished" in result:
        try:
            result["elapsed_seconds"] = (datetime.fromisoformat(result["job_finished"])
                                         - datetime.fromisoformat(result["job_started"])).total_seconds()
        except ValueError:
            pass
    return result
