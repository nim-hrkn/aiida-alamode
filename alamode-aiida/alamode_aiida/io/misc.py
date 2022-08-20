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
