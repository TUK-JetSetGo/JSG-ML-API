"""
이 모듈은 지리적 좌표를 나타내는 값 객체(Coordinate)를 정의하며,
Vincenty's formula를 사용하여 두 좌표 간의 정확한 지오데식 거리를 계산하는 기능을 제공합니다.
"""

import math
from dataclasses import dataclass

# WGS-84 타원체 상수 (미터 단위)
WGS84_A = 6378137.0  # 장반경
WGS84_B = 6356752.314245  # 단반경
WGS84_F = 1 / 298.257223563  # 편평도


@dataclass(frozen=True)
class VincentyResult:
    """Vincenty 공식 반복 계산 결과를 저장하는 값 객체."""

    lambda_val: float
    sigma: float
    sin_sigma: float
    cos_sigma: float
    sin_alpha: float
    cos2_alpha: float
    cos2_sigma_m: float


def _vincenty_iterative(
    flattening: float, trig: tuple, lon_diff: float
) -> VincentyResult:
    """
    Vincenty 공식의 반복 계산을 수행하여 최종 결과를 반환합니다.

    Parameters:
        flattening (float): 편평도.
        trig (tuple): (sin_u1, cos_u1, sin_u2, cos_u2) 값.
        lon_diff (float): 경도 차이 (라디안).

    Returns:
        VincentyResult: 반복 계산 결과.
    """
    sin_u1, cos_u1, sin_u2, cos_u2 = trig
    lambda_val = lon_diff

    def compute_iteration(current_lambda: float):
        s_lam = math.sin(current_lambda)
        c_lam = math.cos(current_lambda)
        sin_sigma = math.sqrt(
            (cos_u2 * s_lam) ** 2 + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * c_lam) ** 2
        )
        if sin_sigma == 0:
            return None  # 두 점이 일치하는 경우
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * c_lam
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cos_u1 * cos_u2 * s_lam / sin_sigma
        cos2_alpha = 1 - sin_alpha**2
        cos2_sigma_m = (
            cos_sigma - 2 * sin_u1 * sin_u2 / cos2_alpha if cos2_alpha != 0 else 0.0
        )
        return sin_sigma, cos_sigma, sigma, sin_alpha, cos2_alpha, cos2_sigma_m

    for _ in range(100):
        prev_lambda = lambda_val
        iteration = compute_iteration(lambda_val)
        if iteration is None:
            return VincentyResult(lambda_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sin_sigma, cos_sigma, sigma, sin_alpha, cos2_alpha, cos2_sigma_m = iteration
        c_value = flattening / 16 * cos2_alpha * (4 + flattening * (4 - 3 * cos2_alpha))
        lambda_val = lon_diff + (1 - c_value) * flattening * sin_alpha * (
            sigma
            + c_value
            * sin_sigma
            * (cos2_sigma_m + c_value * cos_sigma * (-1 + 2 * cos2_sigma_m**2))
        )
        if abs(lambda_val - prev_lambda) < 1e-12:
            return VincentyResult(
                lambda_val,
                sigma,
                sin_sigma,
                cos_sigma,
                sin_alpha,
                cos2_alpha,
                cos2_sigma_m,
            )
    return VincentyResult(
        lambda_val, sigma, sin_sigma, cos_sigma, sin_alpha, cos2_alpha, cos2_sigma_m
    )


def _vincenty_distance(
    semi_major: float, semi_minor: float, result: VincentyResult
) -> float:
    """
    Vincenty 공식 후속 계산으로 두 점 사이의 거리를 계산합니다.

    Parameters:
        semi_major (float): 장반경.
        semi_minor (float): 단반경.
        result (VincentyResult): Vincenty 공식 반복 계산 결과.

    Returns:
        float: 두 좌표 간 거리 (km).
    """
    u2_val = result.cos2_alpha * (semi_major**2 - semi_minor**2) / (semi_minor**2)
    a_factor = 1 + u2_val / 16384 * (
        4096 + u2_val * (-768 + u2_val * (320 - 175 * u2_val))
    )
    b_factor = u2_val / 1024 * (256 + u2_val * (-128 + u2_val * (74 - 47 * u2_val)))
    delta_sigma = (
        b_factor
        * result.sin_sigma
        * (
            result.cos2_sigma_m
            + b_factor
            / 4
            * (
                result.cos_sigma * (-1 + 2 * result.cos2_sigma_m**2)
                - b_factor
                / 6
                * result.cos2_sigma_m
                * (-3 + 4 * result.sin_sigma**2)
                * (-3 + 4 * result.cos2_sigma_m**2)
            )
        )
    )
    distance_m = semi_minor * a_factor * (result.sigma - delta_sigma)
    return distance_m / 1000.0


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    하버사인 공식을 사용하여 두 점 사이의 거리를 계산합니다.

    Parameters:
        lat1 (float): 첫 번째 점의 위도 (라디안).
        lon1 (float): 첫 번째 점의 경도 (라디안).
        lat2 (float): 두 번째 점의 위도 (라디안).
        lon2 (float): 두 번째 점의 경도 (라디안).

    Returns:
        float: 두 좌표 간 거리 (km).
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a_value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c_value = 2 * math.atan2(math.sqrt(a_value), math.sqrt(1 - a_value))
    earth_radius = 6371.0  # km
    return earth_radius * c_value


@dataclass(frozen=True)
class Coordinate:
    """지리적 좌표를 나타내는 값 객체."""

    latitude: float  # 위도
    longitude: float  # 경도

    def distance_to(self, other: "Coordinate") -> float:
        """
        Vincenty's formula를 사용하여 두 좌표 간의 거리를 계산합니다.
        WGS-84 타원체를 기반으로 정확한 지오데식 거리를 반환합니다.

        Parameters:
            other (Coordinate): 비교할 다른 좌표.

        Returns:
            float: 두 좌표 간 거리 (km).
        """
        lat1, lon1, lat2, lon2 = map(
            math.radians,
            [self.latitude, self.longitude, other.latitude, other.longitude],
        )
        u1_angle = math.atan((1 - WGS84_F) * math.tan(lat1))
        u2_angle = math.atan((1 - WGS84_F) * math.tan(lat2))
        trig = (
            math.sin(u1_angle),
            math.cos(u1_angle),
            math.sin(u2_angle),
            math.cos(u2_angle),
        )
        lon_diff = lon2 - lon1

        result = _vincenty_iterative(WGS84_F, trig, lon_diff)
        if result.sigma == 0:
            return 0.0

        distance = _vincenty_distance(WGS84_A, WGS84_B, result)
        if math.isnan(distance):
            return _haversine_distance(lat1, lon1, lat2, lon2)
        return distance

    def is_near(self, other: "Coordinate", threshold_km: float = 1.0) -> bool:
        """
        두 좌표가 지정된 임계값(km) 내에 있는지 확인합니다.

        Parameters:
            other (Coordinate): 비교할 다른 좌표.
            threshold_km (float): 임계값 (km).

        Returns:
            bool: 임계값 내에 있으면 True, 아니면 False.
        """
        return self.distance_to(other) <= threshold_km

    def __str__(self) -> str:
        return f"({self.latitude:.6f}, {self.longitude:.6f})"
