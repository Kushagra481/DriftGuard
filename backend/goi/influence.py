"""GOI-inspired influence in the observable subspace of metric SE(3).

Metric: x=[delta translation / L, delta rotation], L=1 m. Camera-centred
right-local perturbations avoid dependence on arbitrary world translation.
The reported index is a deletion-influence approximation, NOT a GOI paper
reproduction and NOT an outlier classifier. Huber weights use pixel residuals.
"""
import numpy as np
from backend.simulation.geometry import K, camera_points, project, skew


def empty_geometry():
    return {"eigenvalues": [0.]*6, "conditionNumber": 1e12,
            "weakestDirection": [1., 0, 0, 0, 0, 0], "observableRank": 0,
            "translationObservability": 0., "rotationObservability": 0.,
            "coverage": 0., "parallax": 0., "features": [],
            "information": np.zeros((6, 6)), "goiMean": 0.}


def diagnose(landmarks, ids, uv, position, rotation, previous=None):
    if len(ids) < 3:
        return empty_geometry()
    points = camera_points(landmarks[ids], position, rotation)
    predicted, depths = project(landmarks[ids], position, rotation)
    residual_vectors = uv-predicted
    residual = np.linalg.norm(residual_vectors, axis=1)
    # MAD of residual components estimates pixel noise without injected labels.
    components = residual_vectors.reshape(-1)
    sigma = max(.7, float(1.4826*np.median(np.abs(components-np.median(components)))))
    normalized = residual/sigma
    weights = np.minimum(1., 1.345/np.maximum(normalized, 1e-9))
    # Strong outliers classified only by residual; raw influence is independent.
    outliers = normalized > 4.
    weights[outliers] = np.minimum(weights[outliers], .02)
    J = np.zeros((len(ids), 2, 6))
    for i, point in enumerate(points):
        x, y, z = point
        z = max(float(z), .05)
        D = np.array([[K[0, 0]/z, 0, -K[0, 0]*x/z**2],
                      [0, K[1, 1]/z, -K[1, 1]*y/z**2]])
        # Translation measured in units of L=1 m; rotation in radians.
        J[i] = D @ np.column_stack((-np.eye(3), skew(point)))
    per_feature = np.einsum("nai,naj,n->nij", J, J, weights/sigma**2)
    H = per_feature.sum(axis=0)
    eigenvalues, vectors = np.linalg.eigh((H+H.T)*.5)
    eigenvalues = np.maximum(eigenvalues, 0)
    observable = eigenvalues > max(eigenvalues[-1]*1e-6, 1e-8)
    V = vectors[:, observable]
    lambdas = eigenvalues[observable]
    Hinv = (V/np.maximum(lambdas, 1e-12))@V.T if len(lambdas) else np.zeros((6, 6))
    leverage = np.clip(np.einsum("ij,nji->n", Hinv, per_feature), 0, .999)
    gradients = np.einsum("nai,na,n->ni", J, residual_vectors, weights/sigma**2)
    perturbations = gradients @ Hinv.T
    # Influence of deleting observation i: observable metric displacement.
    goi = np.linalg.norm(perturbations, axis=1)/np.maximum(1-leverage, .001)
    occupied = set(zip(np.clip((uv[:, 0]/160).astype(int), 0, 3),
                       np.clip((uv[:, 1]/120).astype(int), 0, 3)))
    coverage = len(occupied)/16
    # Rotation-compensated bearing change, estimated poses only.
    parallax = 0.
    if previous is not None:
        pids, puv, prot = previous
        _, ai, bi = np.intersect1d(ids, pids, return_indices=True)
        if len(ai):
            bearings = np.column_stack(((uv[ai]-[320, 240])/460, np.ones(len(ai))))
            old = np.column_stack(((puv[bi]-[320, 240])/460, np.ones(len(ai))))
            old_in_current = (rotation.inv()*prot).apply(old)
            bearings /= np.linalg.norm(bearings, axis=1)[:, None]
            old_in_current /= np.linalg.norm(old_in_current, axis=1)[:, None]
            parallax = float(np.median(np.arccos(np.clip(np.sum(bearings*old_in_current, axis=1), -1, 1))))
    features = [{
        "id": int(ids[i]), "u": float(uv[i, 0]), "v": float(uv[i, 1]),
        "predictedU": float(predicted[i, 0]), "predictedV": float(predicted[i, 1]),
        "landmark": landmarks[ids[i]].tolist(), "residual": float(residual[i]),
        "weight": float(weights[i]), "goi": float(goi[i]),
        "outlier": bool(outliers[i]), "missing": False,
    } for i in range(len(ids))]
    trace = max(float(np.trace(H)), 1e-12)
    return {
        "eigenvalues": eigenvalues.tolist(),
        "conditionNumber": float(min(eigenvalues[-1]/max(eigenvalues[0], 1e-12), 1e12)),
        "weakestDirection": vectors[:, 0].tolist(), "observableRank": int(observable.sum()),
        "translationObservability": float(np.trace(H[:3, :3])/trace),
        "rotationObservability": float(np.trace(H[3:, 3:])/trace),
        "coverage": coverage, "parallax": parallax, "features": features,
        "information": H, "goiMean": float(goi.mean()),
    }
