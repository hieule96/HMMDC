# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 09:43:38 2021

@author: hieu1
"""
import pdb
from typing import List, Tuple

import numpy as np
from scipy.optimize import curve_fit
import scipy.optimize
import scipy.stats
from skimage.metrics import mean_squared_error
from multiprocessing import Pool
from dataclasses import dataclass, field
import lib.Transform as tf
from lib.Quadtreelib import Image, Node
from typing import TypedDict


class StatisticCalculator:
    """Class for computing various statistical measures."""

    @staticmethod
    def compute_sigma_node_ac(node, LCU):
        """Compute sigma of node AC."""
        img_cu = node.get_points(LCU.img)
        img_dct_cu_AC = tf.dct(img_cu - img_cu.mean())
        sigma = np.std(img_dct_cu_AC)
        return sigma

    @staticmethod
    def get_mse(img1, img2):
        """Compute mean squared error between two images."""
        return mean_squared_error(img1, img2)

    @staticmethod
    def entropy_node(img_dct_cu):
        _, counts = np.unique(img_dct_cu, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    @staticmethod
    def get_rsquare(data_fit, real_data):
        """Compute R-squared statistic."""
        residuals = real_data - data_fit
        ss_tot = np.sum((real_data - np.mean(real_data)) ** 2)
        ss_res = np.sum(residuals ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared


class Rounding:
    @staticmethod
    def round_to_n_decimals(value: float, decimals: int = 3):
        return np.around(value, decimals)


class FunctionCurveFitting:
    @staticmethod
    def negExponential(x, a, b):
        return a * np.exp(b * x)

    @staticmethod
    def exponential(x, a, b, c):
        return a * np.exp(b * x) + c

    @staticmethod
    def cubic(x, a, b, c, d):
        return a * x ** 3 + b * x ** 2 + c * x + d

    @staticmethod
    def linear(x, a, b):
        return a * x + b

    @staticmethod
    def derivateExp(x, a, b):
        return a * b * np.exp(b * x)

    @staticmethod
    def exponential_function2(x, a, b, c, d):
        return a * np.exp(b * x) + c * np.exp(d * x)

    @staticmethod
    def derivateExp2(x, a, b, c, d):
        return a * b * np.exp(b * x) + c * d * np.exp(d * x)

    @staticmethod
    def hyperbole(x, a, b):
        return a * x ** b

    @staticmethod
    def derivatehyperbole(x, a, b):
        return a * b * x ** (b - 1)


@dataclass
class OptimizerParameter:
    lam: dict = field(default_factory=lambda: {
        1: float,
        2: float
    })
    rN: float = field(default=0.01, init=True, repr=True)
    Rt: float = field(default=0.5, init=True, repr=True)
    # Constant QP as the contraint
    _QPMin: int = field(default=0, init=True, repr=False)
    _QPMax: int = field(default=51, init=True, repr=False)
    image: Image = None

    def __post_init__(self):
        if not 0 <= self.rN <= 1:
            raise ValueError(f'rN must be between 0 and 1:{self.rN}')
        self._QPMin = max(self._QPMin, 0)
        self._QPMax = min(self._QPMax, 51)

    @property
    def QPMin(self) -> int:
        return self._QPMin

    @property
    def QPMax(self) -> int:
        return self._QPMax


class OptimizerOutput(TypedDict):
    QP1: List[int]
    QP2: List[int]
    r1: float
    r2: float
    D1: float
    D2: float


class Optimizer:
    def __init__(self, opt_param: OptimizerParameter = None):
        self._globalparam: OptimizerParameter = opt_param

    @property
    def globalparam(self) -> OptimizerParameter:
        return self._globalparam

    @globalparam.setter
    def globalparam(self, new_param: OptimizerParameter):
        self._globalparam: OptimizerParameter = new_param

    @property
    def Rt(self) -> float:
        return self._globalparam.Rt if self._globalparam else None

    @Rt.setter
    def Rt(self, Rt: float):
        if self._globalparam:
            self._globalparam.Rt = Rt

    def initialize_cij(self):
        """
        Parameters
        ----------
        compute_sigma_function: function to compute sigma (with AC Default quantized or not)
        redundancy_ratio: float
            Redundancy parameter to be used alternatively

        Returns
        -------
        sigma : array of float
            Standard deviation of the image
        Ci1 : array of float
            Redundancy parameter of Description 1
        Ci2 : array of float
            Redundancy parameter of Description 2
        """

        for turn, i in enumerate(self.globalparam.image.ctu_list):
            if not i.isSkip:
                if turn % 2 == 0:
                    i.Ci1 = (1.0 - self.globalparam.rN) * self.globalparam.rN
                    i.Ci2 = (1.0 - self.globalparam.rN)
                else:
                    i.Ci1 = (1.0 - self.globalparam.rN)
                    i.Ci2 = (1.0 - self.globalparam.rN) * self.globalparam.rN


class Optimizer_curvefitting(Optimizer):
    FUNCTIONS = {
        "exp3": {
            "curve_fitting_function": FunctionCurveFitting.exponential,
            "derivate_function": FunctionCurveFitting.derivateExp,
            "nbCoeff": 3
        },
        "hyperbolic2": {
            "curve_fitting_function": FunctionCurveFitting.hyperbole,
            "derivate_function": FunctionCurveFitting.derivatehyperbole,
            "nbCoeff": 2
        },
        "exp2": {
            "curve_fitting_function": FunctionCurveFitting.negExponential,
            "derivate_function": FunctionCurveFitting.derivateExp,
            "nbCoeff": 2
        },
        "exp4": {
            "curve_fitting_function": FunctionCurveFitting.exponential_function2,
            "derivate_function": FunctionCurveFitting.derivateExp2,
            "nbCoeff": 4
        },
    }

    def __init__(self, opt_param: OptimizerParameter = None, curve_fitting_function="exp"):
        super().__init__(opt_param)
        self.r_storage = None
        self.function_name = curve_fitting_function
        function_set = self.FUNCTIONS.get(curve_fitting_function)
        if function_set is not None:
            self.curve_fitting_function = function_set['curve_fitting_function']
            self.derivate_function = function_set['derivate_function']
            self.nbCoeff = function_set['nbCoeff']
        else:
            raise ValueError(f"Invalid curve fitting function: {curve_fitting_function}")

    def dj(self, xij: np.ndarray) -> Tuple[float, float]:
        dj = 0
        rj = 0
        i = 0
        for cu in self.globalparam.image.ctu_list:
            # counting only not skip element
            if not cu.isSkip:
                dij = self.curve_fitting_function(xij[i], *cu.DRcoeff)
                dj += dij * cu.aki
                rj += xij[i] * cu.aki
                i += 1
        return dj, rj

    def compteCurveCoefficientMultithread(self):
        with Pool() as p:
            result = p.map(self.curve_fitting, self.globalparam.image.ctu_list)
        for i, cu in enumerate(self.globalparam.image.ctu_list):
            cu.isSkip = result[i][0]
            cu.DRcoeff = result[i][1]
            # Min, Max
            cu.bound = (result[i][3], result[i][2])

    def computeCurveCoefficient(self):
        for ctu in self.globalparam.image.ctu_list:
            isSkip, DRcoeff, MAX_E, MIN_E = self.curve_fitting(ctu)
            ctu.isSkip = isSkip
            ctu.DRcoeff = DRcoeff
            ctu.bound = (MIN_E, MAX_E)

    def compute_block_dct(self, img_cu) -> List[np.ndarray]:
        block8x8_dct = []
        for y in range(0, img_cu.shape[0], 8):
            for x in range(0, img_cu.shape[1], 8):
                block8x8_dct.append([x, y, tf.integerDctTransform8x8(img_cu[y:y + 8, x:x + 8])])
        return block8x8_dct

    def compute_mse_entropy_QP(self, node: Node, QPs) -> dict:
        img_cu = node.get_points(self.globalparam.image.img)
        nbPixel = img_cu.shape[0] * img_cu.shape[1]
        if nbPixel == 0:
            pdb.set_trace()
        if img_cu.std() == 0:
            node.isSkip = True
            return {}
        block8x8_dct = self.compute_block_dct(img_cu)
        result = {QP: {} for QP in QPs}
        for QP in QPs:
            img_rec_2D = np.zeros(img_cu.shape)
            entropy = 0

            for x, y, tu in block8x8_dct:
                tuQ = tf.quant(QP, tu)
                img_dct_cu_Reconstruct = tf.dequant(QP, tuQ)
                entropy += StatisticCalculator.entropy_node(tuQ) * tu.shape[0] * tu.shape[1]
                img_rec_2D[y:y + 8, x:x + 8] = tf.IintegerDctTransform8x8(img_dct_cu_Reconstruct)
                img_rec_2D[img_rec_2D < -128] = -128
                img_rec_2D[img_rec_2D > 127] = 127
            entropy = entropy / nbPixel
            mse = StatisticCalculator.get_mse(img_rec_2D, img_cu)
            result[QP]['entropy'] = entropy
            result[QP]['mse'] = mse
        return result

    def quantCU(QP: int, block8x8_dct: List, nbPixels: int):
        entropy = 0
        ## Trunc because quantificator accept only integer number
        QP = int(QP)
        for _, _, tu in block8x8_dct:
            tuQ = tf.quant(QP, tu)
            entropy += StatisticCalculator.entropy_node(tuQ) * tu.shape[0] * tu.shape[1]
        entropy = entropy / nbPixels

        # Calculate entropy
        return entropy

    def searchQP(QP: int, block8x8_dct: List, rtarget: float, nbPixels: int):
        entropy = Optimizer_curvefitting.quantCU(QP=QP, block8x8_dct=block8x8_dct, nbPixels=nbPixels)
        return entropy - rtarget

    def searchForQPbyRate(self, img: np.ndarray, node: Node, rtarget: float):
        # Take image in partition and dct
        img_cu = node.get_points(img)
        nbPixels = img_cu.shape[0] * img_cu.shape[1]
        block8x8_dct = []
        for y in range(0, img_cu.shape[0], 8):
            for x in range(0, img_cu.shape[1], 8):
                block8x8_dct.append([x, y, tf.integerDctTransform8x8(img_cu[y:y + 8, x:x + 8])])
        # bisect algorithm
        try:
            QP = scipy.optimize.bisect(Optimizer_curvefitting.searchQP, self.globalparam.QPMax, self.globalparam.QPMin,
                                       args=(block8x8_dct, rtarget, nbPixels))
        except ValueError:
            pdb.set_trace()
        entropy = Optimizer_curvefitting.quantCU(QP, block8x8_dct, nbPixels)
        return round(QP), entropy

    def curve_fitting(self, ctu: Node):
        interval = max(1, (self.globalparam.QPMax - self.globalparam.QPMin) // 10)
        QPs = list(range(self.globalparam.QPMin, self.globalparam.QPMax + 1, interval))
        # Ensure QPMax and 51 are included in QPs
        if self.globalparam.QPMax != 51 and self.globalparam.QPMax not in QPs:
            QPs.extend([self.globalparam.QPMax, 51])
        else:
            QPs.append(51)
        # Ensure 10 is included in QPs
        if 10 not in QPs:
            QPs = [10] + QPs
        QPs = np.unique(QPs)
        results = self.compute_mse_entropy_QP(ctu, QPs)
        MAX_E = MIN_E = None
        DRcoeff = None
        if not ctu.isSkip:
            try:
                # Remove entries where mse or entropy is 0
                results = {QP: result for QP, result in results.items() if
                           result['mse'] != 0}
                entropyQP = [result['entropy'] for result in results.values()]
                mseQP = [result['mse'] for result in results.values()]
                # Linear regression, then exponential curve fitting
                if self.function_name == "hyperbolic2":
                    log_entropy = np.log(entropyQP)
                    log_mse = np.log(mseQP)
                    slope, intercept, _, _, _ = scipy.stats.linregress(log_entropy, log_mse)
                    DRcoeff = [np.exp(intercept), slope]
                else:
                    log_mse_QP = np.log(mseQP)
                    slope, intercept, _, _, _ = scipy.stats.linregress(entropyQP, log_mse_QP)
                    DRcoeff = [np.exp(intercept), slope]
                DRcoeff, _ = curve_fit(self.curve_fitting_function, entropyQP, mseQP,
                                       p0=DRcoeff, ftol=0.05, xtol=0.05)
                MAX_E = entropyQP[1] if 10 in QPs else entropyQP[0]
                MIN_E = entropyQP[-1] if self.globalparam.QPMax == 51 else entropyQP[-2]
            except:
                pdb.set_trace()
                ctu.isSkip = True
                DRcoeff = None
                MAX_E = MIN_E = None
        else:
            print("ctu skipped")
        return ctu.isSkip, DRcoeff, MAX_E, MIN_E

    def cost_func(self, r, lam, aki, DR_coeff, Ci) -> float:
        Di = aki * self.curve_fitting_function(r, *DR_coeff)
        Dcost = np.sum(Ci * Di)
        R = np.dot(aki, r)
        cost = Dcost + lam * R
        return cost

    def grad_func(self, r, lam, aki, DR_coeff, Ci) -> np.ndarray:
        # pdb.set_trace()
        Diprime = aki * self.derivate_function(r, *DR_coeff)
        Riprime = aki
        grad = Ci * Diprime + lam * Riprime
        return grad

    def compute_r(self, r, lami, Ci, aki, bounds, DR_coeff_cu) -> np.ndarray:
        result_minize = scipy.optimize.minimize(self.cost_func,
                                                r,
                                                jac=self.grad_func,
                                                args=(lami, aki, DR_coeff_cu, Ci),
                                                bounds=bounds, method='L-BFGS-B')
        assert result_minize.success
        return result_minize.x

    # Direct lambda injection mode
    def contraintRT(self, lami, r_init, Ci, bounds, aki, DR_coeff_cu) -> Tuple[np.ndarray, float]:
        r = self.compute_r(r_init, lami, Ci, aki, bounds, DR_coeff_cu)
        _, ri = self.dj(r)
        return r, ri - self.Rt

    def optimizeQuadtreeLambaCst(self) -> OptimizerOutput:
        self.initialize_cij()
        QP1 = []
        QP2 = []
        DR_coeff_cu = self.globalparam.image.get_DRcoeff()
        DR_coeff_cu = np.hsplit(DR_coeff_cu, self.nbCoeff)
        for i in range(self.nbCoeff):
            DR_coeff_cu[i] = DR_coeff_cu[i].reshape(-1, )
        aki = self.globalparam.image.get_aki()
        bounds = self.globalparam.image.get_bound()
        C1 = self.globalparam.image.get_Ci(1)
        C2 = self.globalparam.image.get_Ci(2)
        self.r_storage = {
            1: self.globalparam.image.get_r(1),
            2: self.globalparam.image.get_r(2)
        }

        def constraintRT_wrapper_1(lam):
            self.r_storage[1], value = self.contraintRT(lam, self.r_storage[1], C1, bounds, aki, DR_coeff_cu)
            return value

        def constraintRt_wrapper_2(lam):
            self.r_storage[2], value = self.contraintRT(lam, self.r_storage[2], C2, bounds, aki, DR_coeff_cu)
            return value

        try:
            self.globalparam.lam[1] = scipy.optimize.bisect(constraintRT_wrapper_1, 0, 1000, rtol=0.01)
        except ValueError:
            # TODO: error handle must be something more sophisticated
            # This is because the bound is not solvable therefore, r will get the minimum QP
            print("Non solvable description 1")
            if constraintRT_wrapper_1(0) < 0 and constraintRT_wrapper_1(1000) < 0:
                self.r_storage[1] = [i[1] for i in bounds]
            else:
                self.r_storage[1] = [i[0] for i in bounds]

        try:
            self.globalparam.lam[2] = scipy.optimize.bisect(constraintRt_wrapper_2, 0, 1000, rtol=0.01)
        except ValueError:
            print("Non solvable description 2")
            if constraintRt_wrapper_2(0) < 0 and constraintRt_wrapper_2(1000) < 0:
                self.r_storage[2] = [i[1] for i in bounds]
            else:
                self.r_storage[2] = [i[0] for i in bounds]
        # Update the value
        self.globalparam.image.set_r(self.r_storage[1], 1)
        self.globalparam.image.set_r(self.r_storage[2], 2)
        entropy1 = []
        entropy2 = []
        for r_count, node in enumerate(self.globalparam.image.ctu_list):
            if node.isSkip:
                QP1.append(self.globalparam.QPMin)
                QP2.append(self.globalparam.QPMin)
            else:
                a, b = self.searchForQPbyRate(self.globalparam.image.img, node, node.r1)
                c, d = self.searchForQPbyRate(self.globalparam.image.img, node, node.r2)
                QP1.append(a)
                entropy1.append(b)
                entropy2.append(d)
                QP2.append(c)
        Di1, ri1 = self.dj(entropy1)
        Di2, ri2 = self.dj(entropy2)
        out: OptimizerOutput = {
            "QP1": QP1,
            "QP2": QP2,
            "D1": Di1,
            "D2": Di2,
            "r1": ri1,
            "r2": ri2,
        }
        return out

    def getRecImg(self, LCU, img, QPs, r):
        img_rec_2D = np.zeros(img.shape)
        i = 0
        mse = 0
        delta = 0
        for cus in LCU.CTUs:
            for node in cus:
                img_cu = node.get_points(img)
                for y in range(0, img_cu.shape[0], 8):
                    for x in range(0, img_cu.shape[1], 8):
                        tu = tf.integerDctTransform8x8(img_cu[y:y + 8, x:x + 8])
                        # pdb.set_trace()
                        tuQ = tf.quant(QPs[i], tu)
                        img_dct_cu_Reconstruct = tf.dequant(QPs[i], tuQ)
                        # Calculate entropy
                        img_rec_2D[node.y0 + y:node.y0 + y + 8,
                        node.x0 + x:node.x0 + x + 8] = tf.IintegerDctTransform8x8(img_dct_cu_Reconstruct)
                        img_rec_2D[img_rec_2D < -128] = -128
                        img_rec_2D[img_rec_2D > 127] = 127
                mse2 = self.curve_fitting_function(r[i], *node.DRcoeff) * node.aki
                mse1 = StatisticCalculator.get_mse(node.get_points(img_rec_2D), img_cu) * node.aki
                delta += abs(mse1 - mse2)
                mse += mse1
                i += 1
        print("Delta Error :", delta)
        img_rec_2D = img_rec_2D + 128
        return img_rec_2D.astype("uint8"), mse
