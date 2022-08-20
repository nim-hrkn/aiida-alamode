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
from aiida.engine import CalcJob


class alamodeBaseCalcjob(CalcJob):
    """base class of alamode for error handling.
    """
    @classmethod
    def define(cls, spec):
        super(alamodeBaseCalcjob, cls).define(spec)

        spec.exit_code(200, "ERROR_NO_RETRIEVED_FOLDER",
                       message="The retrieved folder data node could not be accessed.")

        # stdout file, or some file
        # versus
        # missing, read, parse, imcomplete
        spec.exit_code(302, 'ERROR_OUTPUT_STDOUT_MISSING',
                       message='The retrieved folder did not contain the required stdout output file.')
        spec.exit_code(310, 'ERROR_OUTPUT_STDOUT_READ',
                       message='The stdout output file could not be read.')
        spec.exit_code(311, 'ERROR_OUTPUT_STDOUT_PARSE',
                       message='The stdout output file could not be parsed.')
        spec.exit_code(312, 'ERROR_OUTPUT_STDOUT_INCOMPLETE',
                       message='The stdout output file was incomplete probably because the calculation got interrupted.')

        spec.exit_code(315, 'ERROR_OUTPUT_PATTERN_FILES_MISSING',
                       message='The retrieved folder did not contain the required pattern file.')
        spec.exit_code(316, 'ERROR_OUTPUT_PATTERN_FILES_MISSING',
                       message='The retrieved folder did not contain the required pattern file.')

        spec.exit_code(320, 'ERROR_OUTPUT_STDIN_MISSING',
                       message='The retrieved folder did not contain the required stdout output file.')

        spec.exit_code(325, 'ERROR_OUTPUT_XML_MISSING',
                       message='The retrieved folder did not contain the required xml output file.')
        spec.exit_code(330, 'ERROR_OUTPUT_FCS_MISSING',
                       message='The retrieved folder did not contain the required fcs output file.')
        # general parse

        spec.exit_code(390, 'ERROR_UNEXPECTED_PARSER_EXCEPTION',
                       message='The parser raised an unexpected exception.')
