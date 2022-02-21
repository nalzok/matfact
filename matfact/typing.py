from typing import NewType, TypeAlias

import numpy as np
import numpy.typing as npt


Mat: TypeAlias = npt.NDArray[np.generic]
LowerTriMat = NewType("LowerTriMat", Mat)
EchelonMat = NewType("EchelonMat", Mat)
UpperTriMat = NewType("UpperTriMat", Mat)
PermMat = NewType("PermMat", Mat)
