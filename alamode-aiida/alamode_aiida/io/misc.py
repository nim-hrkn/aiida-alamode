

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
