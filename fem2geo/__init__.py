from fem2geo import model
from fem2geo import plots
from fem2geo.utils import tensor
from fem2geo.utils import transform

import os

dir_testdata = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '..', 'tutorials', 'data'
            )
        )

