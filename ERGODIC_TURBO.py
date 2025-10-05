import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.optimize import differential_evolution
from scipy.interpolate import CubicSpline, BSpline, splprep, splev
from scipy.fft import dctn
import time
from typing import Tuple, Dict, Callable
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import cma
warnings.filterwarnings('ignore')

class TurboErgodicTrajectoryOptimizer:
    """
    TURBO-CHARGED ergodic trajectory optimizer with maximum speed optimizations.
    
    Features:
    - CMA-ES optimization with parallel evaluation
    - Vectorized Fourier coefficient computation
    - Analytic target distribution computation
    - Vectorized robot penalties
    - Multi-resolution optimization
    - Parallel processing on multiple cores
    """
    
    def __init__(self, 
                 domain_bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
                 n_waypoints: int = 100,
                 k_max: int = 6,
                 n_trajectory_samples: int = 1500,
                 s_parameter: float = None,
                 n_cores: int = 10,
                 verbose: bool = True,
                 bspline_degree: int = 3,
                 boundary_penalty: bool = True,
                 ellipse_center: Tuple[float, float] = (0.5, 0.5),
                 ellipse_radius: float = 0.5,
                 ellipse_eccentricity: float = 0.0,
                 bspline_smoothing: float = 0.0,
                 turn_penalty_weight: float = 1.0,
                 sample_initial_from_target: bool = False,
                 robot_penalty_weight: float = 0.01):
        """
        Initialize the turbo-charged ergodic trajectory optimizer.
        """
        self.x_min, self.x_max, self.y_min, self.y_max = domain_bounds
        self.Lx = self.x_max - self.x_min
        self.Ly = self.y_max - self.y_min
        self.M = n_waypoints
        self.Kmax = k_max
        self.n_samples = n_trajectory_samples
        self.n_cores = min(n_cores, mp.cpu_count())
        self.verbose = verbose
        
        # Elliptical workspace parameters (axis-aligned ellipse)
        self.ellipse_center_x, self.ellipse_center_y = ellipse_center
        self.ellipse_a = float(ellipse_radius)
        # For eccentricity e, semi-minor axis b = a * sqrt(1 - e^2)
        self.ellipse_eccentricity = float(np.clip(ellipse_eccentricity, 0.0, 0.999999))
        self.ellipse_b = self.ellipse_a * np.sqrt(max(1e-12, 1.0 - self.ellipse_eccentricity ** 2))
        
        # B-spline degree for trajectory interpolation
        self.bspline_degree = bspline_degree
        # New: smoothing parameter for splprep (higher -> smoother, less sharp turns)
        self.bspline_smoothing = float(max(0.0, bspline_smoothing))
        
        # Boundary penalty flag
        self.boundary_penalty = boundary_penalty
        
        # Turn smoothing weight
        self.turn_penalty_weight = float(max(0.0, turn_penalty_weight))
        
        # Whether to sample initial waypoints from target distribution
        self.sample_initial_from_target = bool(sample_initial_from_target)
        
        # Robot penalty weight (controls balance between ergodicity and smoothness)
        self.robot_penalty_weight = float(max(0.0, robot_penalty_weight))
        
        # Default s parameter for 2D: (2+1)/2 = 1.5
        self.s = s_parameter if s_parameter is not None else 1.5
        
        # Adaptive core allocation for high-resolution cases
        if self.M >= 100:  # High resolution case
            self.n_cores = min(self.n_cores, 16)  # Use more cores for high resolution
            if self.verbose:
                print(f"High resolution detected ({self.M} waypoints), using {self.n_cores} cores")
        
        # Precompute arrays for speed
        self._precompute_arrays()
        
        # Precompute target Fourier coefficients for current target distribution
        self.mu_k = self._compute_target_fourier_coefficients_grid()
        
        # Storage for optimization results
        self.optimization_result = None
        self.optimal_points = None
        
        # Precompute sample times for spline
        self.t_samples = np.linspace(0, 1, self.n_samples)
        
        if self.verbose:
            print(f"Turbo optimizer initialized with {self.n_cores} cores")
            print(f"Domain: [{self.x_min:.1f}, {self.x_max:.1f}] × [{self.y_min:.1f}, {self.y_max:.1f}] (for grids/basis)")
            print(f"Workspace shape: Ellipse(center=({self.ellipse_center_x:.2f},{self.ellipse_center_y:.2f}), a={self.ellipse_a:.3f}, e={self.ellipse_eccentricity:.3f}, b={self.ellipse_b:.3f})")
            print(f"Waypoints: {self.M}, Fourier modes: {self.Kmax}, Samples: {self.n_samples}")
            print(f"B-spline degree: {self.bspline_degree}")
            if self.bspline_smoothing > 0.0:
                print(f"B-spline smoothing: {self.bspline_smoothing}")
            if self.sample_initial_from_target:
                print("Initial waypoints: grid-based importance sampling from target distribution")
            print(f"Robot penalty weight: {self.robot_penalty_weight} (ergodicity vs smoothness balance)")
    
    def _precompute_arrays(self):
        """Precompute arrays for vectorized operations."""
        # Precompute k arrays for broadcasting
        self.k_array = np.arange(self.Kmax + 1)
        
        # Precompute weights matrix (vectorized)
        k1_sq = self.k_array[:, None]**2  # (K+1, 1)
        k2_sq = self.k_array[None, :]**2  # (1, K+1)
        k_norm_sq = k1_sq + k2_sq  # (K+1, K+1)
        self.Lambda_k = 1.0 / (1 + k_norm_sq)**self.s
        
        # Precompute pi/L factors for cosine evaluation
        self.pi_over_Lx = np.pi / self.Lx
        self.pi_over_Ly = np.pi / self.Ly
    
    def _compute_target_fourier_coefficients_turbo(self) -> np.ndarray:
        """Kept for API compatibility; uses grid-based DCT for arbitrary shapes (ellipse)."""
        return self._compute_target_fourier_coefficients_grid()
    
    def _compute_target_fourier_coefficients_grid(self, n=128) -> np.ndarray:
        """
        FAST grid-based computation for arbitrary target distributions.
        Uses DCT instead of double integration.
        """
        if self.verbose:
            print("Computing target Fourier coefficients (grid-based DCT)...")
        
        x = np.linspace(self.x_min, self.x_max, n)
        y = np.linspace(self.y_min, self.y_max, n)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Compute target distribution on grid
        MU = self._target_distribution(X, Y)
        
        # DCT-II approximates Neumann cosine series
        C = dctn(MU, type=2, norm='ortho')
        
        # Keep only [0..Kmax]×[0..Kmax]
        mu = C[:self.Kmax+1, :self.Kmax+1]
        
        # Normalize so that mu[0,0] ≈ 1
        mu /= mu[0, 0]
        
        if self.verbose:
            print("Target Fourier coefficients computed (DCT)!")
        
        return mu
    
    def _target_distribution(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Uniform over ellipse (center, a, b), zero outside, normalized to 1."""
        dx = (x - self.ellipse_center_x) / (self.ellipse_a + 1e-12)
        dy = (y - self.ellipse_center_y) / (self.ellipse_b + 1e-12)
        r2 = dx * dx + dy * dy
        inside = (r2 <= 1.0).astype(float)
        area = max(np.pi * self.ellipse_a * self.ellipse_b, 1e-12)
        return inside / area

    def _project_points_to_ellipse(self, points: np.ndarray) -> np.ndarray:
        """Project any points outside the ellipse onto its boundary (axis-aligned)."""
        px = points[:, 0]
        py = points[:, 1]
        dx = px - self.ellipse_center_x
        dy = py - self.ellipse_center_y
        nx = dx / (self.ellipse_a + 1e-12)
        ny = dy / (self.ellipse_b + 1e-12)
        r2 = nx * nx + ny * ny
        outside = r2 > 1.0
        if np.any(outside):
            scale = 1.0 / np.sqrt(r2[outside])
            nx_out = nx[outside] * scale
            ny_out = ny[outside] * scale
            px[outside] = self.ellipse_center_x + nx_out * self.ellipse_a
            py[outside] = self.ellipse_center_y + ny_out * self.ellipse_b
        projected = np.column_stack((px, py))
        return projected
    
    def _create_spline_trajectory_turbo(self, points: np.ndarray) -> np.ndarray:
        """
        ROBUST B-spline trajectory creation with comprehensive error handling.
        B-splines stay within the convex hull of control points, ensuring no overshoot.
        """
        # Ensure we have enough points for spline interpolation
        if len(points) < 4:
            return self._create_linear_trajectory_turbo(points)
        
        # Ensure input points are within ellipse
        points_bounded = self._project_points_to_ellipse(np.copy(points))
        
        # Clean up points to avoid B-spline failures
        points_clean = self._clean_points_for_bspline(points_bounded)
        
        try:
            # Use B-spline interpolation with user-specified degree
            tck, u = splprep([points_clean[:, 0], points_clean[:, 1]], 
                           s=self.bspline_smoothing,  # smoothing (0 = interpolate all points)
                           k=min(self.bspline_degree, len(points_clean) - 1),  # User-specified degree
                           per=False)  # Not periodic
            
            # Evaluate B-spline at sample points
            u_samples = np.linspace(0, 1, self.n_samples)
            x_vals, y_vals = splev(u_samples, tck)
            
            trajectory = np.column_stack([x_vals, y_vals])
            
            # Project any out-of-ellipse samples back to the ellipse
            trajectory = self._project_points_to_ellipse(trajectory)
            
            # Additional check: if trajectory is too close to boundaries, use linear interpolation
            x_range = trajectory[:, 0].max() - trajectory[:, 0].min()
            y_range = trajectory[:, 1].max() - trajectory[:, 1].min()
            
            # If trajectory doesn't use enough of the domain, use linear interpolation
            if x_range < 0.3 * (self.Lx) or y_range < 0.3 * (self.Ly):
                if self.verbose:
                    print("Warning: B-spline trajectory too constrained, using linear interpolation")
                trajectory = self._create_linear_trajectory_turbo(points_bounded)
            
        except Exception as e:
            # Fallback to linear interpolation
            trajectory = self._create_linear_trajectory_turbo(points_bounded)
        
        # Ensure final trajectory is within ellipse
        return self._project_points_to_ellipse(trajectory)
    
    def _clean_points_for_bspline(self, points: np.ndarray) -> np.ndarray:
        """
        Clean points to avoid common B-spline failures:
        - Remove duplicate or nearly duplicate points
        - Ensure minimum distance between consecutive points
        - Handle collinear points
        """
        if len(points) < 2:
            return points
        
        # Remove duplicate points with tolerance
        tolerance = 1e-6
        cleaned_points = [points[0]]
        
        for i in range(1, len(points)):
            # Check distance from last added point
            dist = np.linalg.norm(points[i] - cleaned_points[-1])
            if dist > tolerance:
                cleaned_points.append(points[i])
        
        cleaned_points = np.array(cleaned_points)
        
        # Ensure we have at least 2 points
        if len(cleaned_points) < 2:
            # If all points were duplicates, create a small line
            cleaned_points = np.array([
                points[0],
                points[0] + np.array([tolerance * 10, tolerance * 10])
            ])
        
        # If we have exactly 2 or 3 points, add intermediate points to avoid degeneracy
        if len(cleaned_points) == 2:
            # Add a midpoint with slight perturbation
            midpoint = (cleaned_points[0] + cleaned_points[1]) / 2
            # Add small perpendicular offset to avoid collinearity
            direction = cleaned_points[1] - cleaned_points[0]
            perp = np.array([-direction[1], direction[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-10) * tolerance * 10
            
            cleaned_points = np.array([
                cleaned_points[0],
                midpoint + perp,
                cleaned_points[1]
            ])
        
        elif len(cleaned_points) == 3:
            # Check if points are collinear and add perturbation if needed
            v1 = cleaned_points[1] - cleaned_points[0]
            v2 = cleaned_points[2] - cleaned_points[1]
            
            # Cross product in 2D (z-component)
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            
            if abs(cross) < tolerance:
                # Points are nearly collinear, add perturbation to middle point
                direction = cleaned_points[2] - cleaned_points[0]
                perp = np.array([-direction[1], direction[0]])
                perp = perp / (np.linalg.norm(perp) + 1e-10) * tolerance * 10
                cleaned_points[1] += perp
        
        # Ensure points lie within ellipse after cleaning
        return self._project_points_to_ellipse(cleaned_points)
    
    def _create_linear_trajectory_turbo(self, points: np.ndarray) -> np.ndarray:
        """FAST linear trajectory creation."""
        # Use precomputed sample times
        t = self.t_samples
        
        # Segment indices and interpolation weights (vectorized)
        segment_ids = (t * (self.M - 1)).astype(int)
        alpha = (t * (self.M - 1)) - segment_ids
        
        # Ensure we don't go out of bounds
        segment_ids = np.clip(segment_ids, 0, self.M - 2)
        
        # Linear interpolation (vectorized)
        P0 = points[segment_ids]
        P1 = points[segment_ids + 1]
        trajectory = (1 - alpha)[:, None] * P0 + alpha[:, None] * P1
        
        # Ensure trajectory lies within ellipse
        return self._project_points_to_ellipse(trajectory)
    
    def _compute_trajectory_fourier_coefficients_turbo(self, trajectory: np.ndarray) -> np.ndarray:
        """
        VECTORIZED Fourier coefficient computation.
        Computes all modes at once instead of nested loops.
        """
        x, y = trajectory[:, 0], trajectory[:, 1]  # (N,)
        
        # Broadcasted cosine computation for all k1 and k2
        # (N, K+1): broadcasted cos for all k1 and k2
        CX = np.cos(self.pi_over_Lx * np.outer(x, self.k_array))
        CY = np.cos(self.pi_over_Ly * np.outer(y, self.k_array))
        
        # (K+1, K+1): mean over N of outer products
        # Each row of CX corresponds to a sample; same for CY
        # mean(CX_i[:,None] * CY_i[None,:]) over i:
        C = (CX[:, :, None] * CY[:, None, :]).mean(axis=0)
        
        return C  # 2D array (K+1, K+1)
    
    def _calculate_robot_penalty_turbo(self, trajectory: np.ndarray) -> float:
        """
        VECTORIZED robot motion penalty computation.
        No Python loops - everything vectorized.
        """
        if len(trajectory) < 2:
            return 0.0
        
        # Compute all derivatives in one pass
        v = np.diff(trajectory, axis=0)  # (N-1, 2)
        a = np.diff(v, axis=0) if len(v) > 1 else np.array([])  # (N-2, 2)
        j = np.diff(a, axis=0) if len(a) > 1 else np.array([])  # (N-3, 2)
        
        # Compute magnitudes (vectorized)
        vm = np.linalg.norm(v, axis=1)
        am = np.linalg.norm(a, axis=1) if len(a) > 0 else np.array([])
        jm = np.linalg.norm(j, axis=1) if len(j) > 0 else np.array([])
        
        # Curvature computation (vectorized)
        if len(v) > 1:
            v1 = v[:-1]
            v2 = v[1:]
            # 2D cross product magnitude = |x1*y2 - y1*x2|
            cross = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
            denom = (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
            curv = np.where(denom > 1e-12, cross / denom, 0.0)
            # Turning angle in radians between consecutive segments
            dot = (v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1])
            norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
            cos_theta = np.clip(np.where(norms > 1e-12, dot / norms, 1.0), -1.0, 1.0)
            turn_angles = np.arccos(cos_theta)
        else:
            curv = np.array([])
            turn_angles = np.array([])
        
        # Compute penalties (vectorized)
        velocity_penalty = vm.mean() / 0.1 if len(vm) > 0 else 0.0
        acceleration_penalty = am.mean() / 0.01 if len(am) > 0 else 0.0
        jerk_penalty = jm.mean() / 0.001 if len(jm) > 0 else 0.0
        curvature_penalty = curv.mean() / 0.08 if len(curv) > 0 else 0.0  # stronger baseline
        curvature_sq_penalty = (np.mean(curv**2) / 0.02) if len(curv) > 0 else 0.0
        sharp_turn_penalty = (np.mean((turn_angles**2)) / (np.pi/6)**2) if len(turn_angles) > 0 else 0.0
        length_penalty = vm.sum() / 2.0 if len(vm) > 0 else 0.0
        
        # Initialize boundary penalties
        boundary_penalty = 0.0
        violation_penalty = 0.0
        
        # Boundary proximity penalty based on ellipse
        if self.boundary_penalty:
            nx = (trajectory[:, 0] - self.ellipse_center_x) / (self.ellipse_a + 1e-12)
            ny = (trajectory[:, 1] - self.ellipse_center_y) / (self.ellipse_b + 1e-12)
            r = np.sqrt(nx * nx + ny * ny)
            margin = 0.05
            boundary_penalty = np.maximum(0.0, (r - (1.0 - margin)) / margin).mean()
            violation_penalty = np.mean((r > 1.0).astype(float)) * 10.0
        
        # Combine penalties
        penalty = (0.5 * velocity_penalty + 
                  1.0 * acceleration_penalty + 
                  0.5 * jerk_penalty + 
                  (1.5 * curvature_penalty + 1.0 * curvature_sq_penalty + self.turn_penalty_weight * sharp_turn_penalty) + 
                  0.3 * length_penalty + 
                  2.0 * boundary_penalty + 
                  5.0 * violation_penalty)
        
        return penalty
    
    def ergodicity_cost_turbo(self, points_flat: np.ndarray) -> float:
        """
        TURBO-CHARGED cost function with vectorized computation.
        """
        # Reshape to (M, 2)
        points = points_flat.reshape(self.M, 2)
        
        # Create the full trajectory using splines
        trajectory = self._create_spline_trajectory_turbo(points)
        
        # Compute Fourier coefficients (vectorized)
        C = self._compute_trajectory_fourier_coefficients_turbo(trajectory)
        
        # Compute ergodicity metric (vectorized)
        diff = C - self.mu_k
        phi2 = np.sum(self.Lambda_k * diff * diff)
        
        # Add robot penalty
        robot_penalty = self._calculate_robot_penalty_turbo(trajectory)
        
        # Combined cost with configurable robot penalty weight
        total_cost = phi2 + self.robot_penalty_weight * robot_penalty
        
        return total_cost
    
    def _parallel_cost_evaluation(self, points_list):
        """Parallel cost evaluation for CMA-ES."""
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            costs = list(executor.map(self.ergodicity_cost_turbo, points_list))
        return costs
    
    def _sample_points_from_target(self, num_points: int, grid_n: int = 256) -> np.ndarray:
        """Sample points from the current target distribution using grid-based importance sampling."""
        # Build grid over domain
        x = np.linspace(self.x_min, self.x_max, grid_n)
        y = np.linspace(self.y_min, self.y_max, grid_n)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Evaluate target density on grid
        density = self._target_distribution(X, Y)
        density = np.maximum(density, 0.0)
        flat = density.ravel()
        total = flat.sum()
        if total <= 0:
            # Fallback to uniform in ellipse
            pts = np.random.uniform([self.x_min, self.y_min], [self.x_max, self.y_max], size=(num_points, 2))
            return self._project_points_to_ellipse(pts)
        
        p = flat / total
        idx = np.random.choice(flat.size, size=num_points, replace=True, p=p)
        i = idx // grid_n
        j = idx % grid_n
        
        # Map to coordinates and add small jitter within cell
        dx = (self.x_max - self.x_min) / max(1, grid_n - 1)
        dy = (self.y_max - self.y_min) / max(1, grid_n - 1)
        jitter_x = (np.random.rand(num_points) - 0.5) * dx
        jitter_y = (np.random.rand(num_points) - 0.5) * dy
        sampled_x = x[i] + jitter_x
        sampled_y = y[j] + jitter_y
        points = np.column_stack((sampled_x, sampled_y))
        
        # Ensure points lie within ellipse
        return self._project_points_to_ellipse(points)

    def optimize_turbo(self, 
                      initial_points: np.ndarray = None,
                      max_iterations: int = 100,
                      population_size: int = None,
                      seed: int = 42) -> Dict:
        """
        TURBO-CHARGED optimization using CMA-ES with parallel evaluation.
        """
        if self.verbose:
            print(f"Starting TURBO-CHARGED ergodic trajectory optimization...")
            print(f"  Using CMA-ES with {self.n_cores} parallel cores")
            print(f"  Domain: [{self.x_min:.1f}, {self.x_max:.1f}] × [{self.y_min:.1f}, {self.y_max:.1f}] (for grids/basis)")
            print(f"  Workspace: Ellipse(center=({self.ellipse_center_x:.2f},{self.ellipse_center_y:.2f}), a={self.ellipse_a:.3f}, b={self.ellipse_b:.3f})")
            print(f"  Waypoints: {self.M}")
            print(f"  Fourier modes: {self.Kmax}")
            print(f"  Trajectory samples: {self.n_samples}")
        
        # Set random seed
        np.random.seed(seed)
        
        # Define rectangular bounds over domain; ellipse constraint handled by projection/penalty
        bounds = [(self.x_min, self.x_max)] * self.M + [(self.y_min, self.y_max)] * self.M
        
        # Initial points (random if not provided)
        if initial_points is None:
            if self.sample_initial_from_target:
                initial_points = self._sample_points_from_target(self.M)
            else:
                initial_points = np.random.uniform(
                    [self.x_min, self.y_min], 
                    [self.x_max, self.y_max], 
                    (self.M, 2)
                )
        # Project to ellipse
        initial_points = self._project_points_to_ellipse(initial_points)
        
        # Flatten initial points
        x0 = initial_points.flatten()
        
        # Set CMA-ES parameters with adaptive sizing for high-dimensional problems
        if population_size is None:
            if len(x0) >= 200:  # High-dimensional case (100 waypoints * 2 coordinates)
                population_size = 8 + int(4 * np.log(len(x0)))  # Larger population for high-D
            else:
                population_size = 4 + int(3 * np.log(len(x0)))  # CMA-ES default
        
        # Define bounds for CMA-ES
        bounds_array = np.array(bounds)
        xmin = bounds_array[:, 0]
        xmax = bounds_array[:, 1]
        
        # Run CMA-ES optimization
        start_time = time.time()
        
        # Create CMA-ES optimizer
        es = cma.CMAEvolutionStrategy(
            x0, 
            0.5,  # Initial sigma
            {
                'maxiter': max_iterations,
                'popsize': population_size,
                'seed': seed,
                'verbose': 0 if self.verbose else -1,
                'CMA_diagonal': False,  # Use full covariance matrix
                'CMA_elitist': True,    # Elitist selection
            }
        )
        
        # Run optimization with thread-based parallelism (avoids pickling issues)
        while not es.stop():
            solutions = es.ask()
            
            # Use ThreadPoolExecutor to avoid pickling issues
            if self.n_cores > 1 and len(solutions) >= 4:
                try:
                    with ThreadPoolExecutor(max_workers=self.n_cores) as executor:
                        fitness_values = list(executor.map(self.ergodicity_cost_turbo, solutions))
                except Exception as e:
                    if self.verbose:
                        print(f"Parallel execution failed: {e}. Using sequential.")
                    fitness_values = [self.ergodicity_cost_turbo(x) for x in solutions]
            else:
                # Sequential evaluation for small populations or single core
                fitness_values = [self.ergodicity_cost_turbo(x) for x in solutions]
            
            es.tell(solutions, fitness_values)
            
            # Enhanced progress tracking for longer runs
            if self.verbose:
                if es.countiter % 10 == 0 or es.countiter <= 5:
                    best_fitness = min(fitness_values)
                    avg_fitness = np.mean(fitness_values)
                    print(f"Generation {es.countiter:3d}: Best = {best_fitness:.6f}, Avg = {avg_fitness:.6f}")
                elif es.countiter % 50 == 0:
                    best_fitness = min(fitness_values)
                    print(f"Generation {es.countiter:3d}: Best fitness = {best_fitness:.6f} (checkpoint)")
        
        end_time = time.time()
        
        # Get best solution
        best_x = es.result.xbest
        best_fitness = es.result.fbest
        
        # Store results
        self.optimization_result = {
            'x': best_x,
            'fun': best_fitness,
            'generations': es.countiter,
            'evaluations': es.result.evaluations
        }
        self.optimal_points = best_x.reshape(self.M, 2)
        
        if self.verbose:
            print(f"\nTURBO optimization completed!")
            print(f"  Final fitness: {best_fitness:.6f}")
            print(f"  Generations: {es.countiter}")
            print(f"  Function evaluations: {es.result.evaluations}")
            print(f"  Optimization time: {end_time - start_time:.2f} seconds")
            print(f"  Speed: {es.result.evaluations / (end_time - start_time):.1f} evals/sec")
        
        # Analyze the final solution
        final_trajectory = self._create_spline_trajectory_turbo(self.optimal_points)
        ergodicity_cost = self._calculate_ergodicity_cost_only_turbo(final_trajectory)
        robot_penalty = self._calculate_robot_penalty_turbo(final_trajectory)
        
        if self.verbose:
            print(f"  Ergodicity cost: {ergodicity_cost:.6f}")
            print(f"  Robot penalty: {robot_penalty:.6f}")
        
        return {
            'optimal_points': self.optimal_points,
            'total_cost': best_fitness,
            'ergodicity_cost': ergodicity_cost,
            'robot_penalty': robot_penalty,
            'optimization_time': end_time - start_time,
            'function_evaluations': es.result.evaluations,
            'generations': es.countiter,
            'initial_points': initial_points
        }
    
    def _calculate_ergodicity_cost_only_turbo(self, trajectory: np.ndarray) -> float:
        """Calculate only the ergodicity cost without robot penalty."""
        C = self._compute_trajectory_fourier_coefficients_turbo(trajectory)
        diff = C - self.mu_k
        phi2 = np.sum(self.Lambda_k * diff * diff)
        return phi2
    
    def optimize_multi_resolution(self, 
                                initial_points: np.ndarray = None,
                                seed: int = 42) -> Dict:
        """
        Multi-resolution optimization: coarse-to-fine approach.
        """
        if self.verbose:
            print("Starting multi-resolution optimization...")
        
        # Calculate k_max values for each stage based on main k_max
        coarse_kmax = max(2, self.Kmax // 2)  # Half of main k_max, minimum 2
        medium_kmax = max(3, int(self.Kmax * 0.75))  # 75% of main k_max, minimum 3
        fine_kmax = self.Kmax  # Full resolution
        
        # Stage 1: Coarse optimization
        if self.verbose:
            print(f"Stage 1: Coarse optimization (Kmax={coarse_kmax}, n_samples=500)")
        
        # Create coarse optimizer
        coarse_optimizer = TurboErgodicTrajectoryOptimizer(
            domain_bounds=(self.x_min, self.x_max, self.y_min, self.y_max),
            n_waypoints=self.M,
            k_max=coarse_kmax,
            n_trajectory_samples=500,
            s_parameter=self.s,
            n_cores=self.n_cores,
            verbose=False,
            bspline_degree=self.bspline_degree,
            ellipse_center=(self.ellipse_center_x, self.ellipse_center_y),
            ellipse_radius=self.ellipse_a,
            ellipse_eccentricity=self.ellipse_eccentricity
        )
        
        # Run coarse optimization
        coarse_result = coarse_optimizer.optimize_turbo(
            initial_points=initial_points,
            max_iterations=50,
            seed=seed
        )
        
        if self.verbose:
            print(f"Coarse optimization completed. Best cost: {coarse_result['total_cost']:.6f}")
        
        # Stage 2: Fine optimization (warm start from coarse result)
        if self.verbose:
            print(f"Stage 2: Fine optimization (Kmax={fine_kmax}, n_samples=1500)")
        
        # Run fine optimization with warm start
        fine_result = self.optimize_turbo(
            initial_points=coarse_result['optimal_points'],
            max_iterations=50,
            seed=seed + 1
        )
        
        if self.verbose:
            print(f"Fine optimization completed. Best cost: {fine_result['total_cost']:.6f}")
            improvement = ((coarse_result['total_cost'] - fine_result['total_cost']) / 
                          coarse_result['total_cost'] * 100)
            print(f"Improvement: {improvement:.2f}%")
        
        return fine_result
    
    def optimize_multi_resolution_extended(self, 
                                         initial_points: np.ndarray = None,
                                         seed: int = 42) -> Dict:
        """
        EXTENDED multi-resolution optimization: coarse-to-fine approach with MORE iterations.
        Designed for higher resolution (100 waypoints) and longer optimization.
        """
        if self.verbose:
            print("Starting EXTENDED multi-resolution optimization...")
            print("  Target: 100 waypoints with extended optimization time")
        
        # Calculate k_max values for each stage based on main k_max
        coarse_kmax = max(2, self.Kmax // 2)  # Half of main k_max, minimum 2
        medium_kmax = max(3, int(self.Kmax * 0.75))  # 75% of main k_max, minimum 3
        fine_kmax = self.Kmax  # Full resolution
        
        # Stage 1: Coarse optimization (longer)
        if self.verbose:
            print(f"Stage 1: Coarse optimization (Kmax={coarse_kmax}, n_samples=1000, 100 iterations)")
        
        # Create coarse optimizer with more samples
        coarse_optimizer = TurboErgodicTrajectoryOptimizer(
            domain_bounds=(self.x_min, self.x_max, self.y_min, self.y_max),
            n_waypoints=self.M,
            k_max=coarse_kmax,
            n_trajectory_samples=1000,  # Increased from 500
            s_parameter=self.s,
            n_cores=self.n_cores,
            verbose=False,
            bspline_degree=self.bspline_degree,
            ellipse_center=(self.ellipse_center_x, self.ellipse_center_y),
            ellipse_radius=self.ellipse_a,
            ellipse_eccentricity=self.ellipse_eccentricity
        )
        
        # Run coarse optimization with MORE iterations
        coarse_result = coarse_optimizer.optimize_turbo(
            initial_points=initial_points,
            max_iterations=100,  # DOUBLED from 50 to 100
            seed=seed
        )
        
        if self.verbose:
            print(f"Coarse optimization completed. Best cost: {coarse_result['total_cost']:.6f}")
        
        # Stage 2: Medium optimization (new stage)
        if self.verbose:
            print(f"Stage 2: Medium optimization (Kmax={medium_kmax}, n_samples=1500, 150 iterations)")
        
        # Create medium optimizer
        medium_optimizer = TurboErgodicTrajectoryOptimizer(
            domain_bounds=(self.x_min, self.x_max, self.y_min, self.y_max),
            n_waypoints=self.M,
            k_max=medium_kmax,  # Scaled based on main k_max
            n_trajectory_samples=1500,
            s_parameter=self.s,
            n_cores=self.n_cores,
            verbose=False,
            bspline_degree=self.bspline_degree,
            ellipse_center=(self.ellipse_center_x, self.ellipse_center_y),
            ellipse_radius=self.ellipse_a,
            ellipse_eccentricity=self.ellipse_eccentricity
        )
        
        # Run medium optimization
        medium_result = medium_optimizer.optimize_turbo(
            initial_points=coarse_result['optimal_points'],
            max_iterations=150,  # 150 iterations for medium stage
            seed=seed + 1
        )
        
        if self.verbose:
            print(f"Medium optimization completed. Best cost: {medium_result['total_cost']:.6f}")
            improvement = ((coarse_result['total_cost'] - medium_result['total_cost']) / 
                          coarse_result['total_cost'] * 100)
            print(f"Improvement from coarse: {improvement:.2f}%")
        
        # Stage 3: Fine optimization (warm start from medium result)
        if self.verbose:
            print(f"Stage 3: Fine optimization (Kmax={fine_kmax}, n_samples=2000, 200 iterations)")
        
        # Run fine optimization with warm start and MORE iterations
        fine_result = self.optimize_turbo(
            initial_points=medium_result['optimal_points'],
            max_iterations=200,  # QUADRUPLED from 50 to 200
            seed=seed + 2
        )
        
        if self.verbose:
            print(f"Fine optimization completed. Best cost: {fine_result['total_cost']:.6f}")
            improvement = ((medium_result['total_cost'] - fine_result['total_cost']) / 
                          medium_result['total_cost'] * 100)
            print(f"Improvement from medium: {improvement:.2f}%")
            
            total_improvement = ((coarse_result['total_cost'] - fine_result['total_cost']) / 
                               coarse_result['total_cost'] * 100)
            print(f"Total improvement: {total_improvement:.2f}%")
        
        return fine_result
    
    def visualize_results(self, 
                         show_target: bool = True,
                         show_trajectory: bool = True,
                         show_waypoints: bool = True,
                         n_grid_points: int = 100):
        """
        Visualize the optimization results.
        """
        if self.optimal_points is None:
            print("No optimization results to visualize. Run optimize_turbo() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Create grid for visualization
        x_grid = np.linspace(self.x_min, self.x_max, n_grid_points)
        y_grid = np.linspace(self.y_min, self.y_max, n_grid_points)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Plot 1: Target distribution
        if show_target:
            target_vals = self._target_distribution(X, Y)
            im1 = ax1.contourf(X, Y, target_vals, levels=20, cmap='viridis')
            ax1.set_title('Target Distribution μ(x,y)')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            plt.colorbar(im1, ax=ax1)
        
        # Plot 2: Optimized trajectory with clear color cycling
        if show_trajectory:
            trajectory = self._create_spline_trajectory_turbo(self.optimal_points)

            # Use a categorical palette with high contrast
            palette = list(plt.get_cmap('tab20').colors)
            n_segments = 20
            segment_len = max(2, len(trajectory) // n_segments)

            label_added = False
            for start_idx in range(0, len(trajectory) - 1, segment_len):
                end_idx = min(len(trajectory), start_idx + segment_len + 1)
                segment = trajectory[start_idx:end_idx]
                color = palette[(start_idx // segment_len) % len(palette)]
                if not label_added:
                    ax2.plot(segment[:, 0], segment[:, 1], color=color, linewidth=2, alpha=0.9, label='B-Spline Trajectory')
                    label_added = True
                else:
                    ax2.plot(segment[:, 0], segment[:, 1], color=color, linewidth=2, alpha=0.9)
        
        if show_waypoints:
            ax2.plot(self.optimal_points[:, 0], self.optimal_points[:, 1], 'ro', 
                    markersize=8, label='Waypoints')
            ax2.plot(self.optimal_points[:, 0], self.optimal_points[:, 1], 'r--', 
                    linewidth=1, alpha=0.5)
        
        ax2.set_title(f'TURBO B-Spline Ergodic Trajectory\n(Total Cost: {self.optimization_result["fun"]:.6f})')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add ellipse workspace visualization
        from matplotlib.patches import Ellipse
        ellipse_patch = Ellipse((self.ellipse_center_x, self.ellipse_center_y),
                                2 * self.ellipse_a, 2 * self.ellipse_b,
                                fill=False, edgecolor='green', linestyle='--', linewidth=2, alpha=0.7)
        ax2.add_patch(ellipse_patch)
        ax2.text(0.02, 0.98, f'Ellipse a={self.ellipse_a:.2f}, b={self.ellipse_b:.2f}', 
                transform=ax2.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
        
        # Set consistent axis limits
        for ax in [ax1, ax2]:
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(self.y_min, self.y_max)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_ergodicity(self):
        """
        Analyze the ergodicity of the optimized trajectory.
        """
        if self.optimal_points is None:
            print("No optimization results to analyze. Run optimize_turbo() first.")
            return
        
        print("\n=== TURBO Ergodicity Analysis ===")
        
        # Get the full trajectory
        trajectory = self._create_spline_trajectory_turbo(self.optimal_points)
        
        # Compute trajectory Fourier coefficients
        C = self._compute_trajectory_fourier_coefficients_turbo(trajectory)
        
        # Compare with target coefficients
        print("\nFourier Coefficient Comparison (TURBO B-Spline Trajectory):")
        print("Mode (k1,k2) | Target μ_k | Trajectory c_k | Difference | Weight")
        print("-" * 65)
        
        total_weighted_error = 0
        for k1 in range(self.Kmax + 1):
            for k2 in range(self.Kmax + 1):
                mu_val = self.mu_k[k1, k2]
                c_val = C[k1, k2]
                diff = c_val - mu_val
                
                Lambda = self.Lambda_k[k1, k2]
                weighted_error = Lambda * diff**2
                total_weighted_error += weighted_error
                
                print(f"({k1:2d},{k2:2d})      | {mu_val:8.4f} | {c_val:12.4f} | {diff:9.4f} | {Lambda:5.3f}")
        
        print("-" * 65)
        print(f"Total weighted error: {total_weighted_error:.6f}")
        
        # Calculate costs
        ergodicity_cost = self._calculate_ergodicity_cost_only_turbo(trajectory)
        robot_penalty = self._calculate_robot_penalty_turbo(trajectory)
        total_cost = ergodicity_cost + 0.1 * robot_penalty
        
        print(f"Ergodicity cost: {ergodicity_cost:.6f}")
        print(f"Robot penalty: {robot_penalty:.6f}")
        print(f"Total cost: {total_cost:.6f}")
        
        # Performance metrics
        print(f"\nPerformance Analysis:")
        print(f"  Total trajectory points: {len(trajectory)}")
        print(f"  Trajectory length: {np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)):.4f}")
        
        # Smoothness metrics
        velocities = np.diff(trajectory, axis=0)
        accelerations = np.diff(velocities, axis=0) if len(velocities) > 1 else np.array([])
        
        print(f"  Average velocity: {np.mean(np.linalg.norm(velocities, axis=1)):.6f}")
        if len(accelerations) > 0:
            print(f"  Average acceleration: {np.mean(np.linalg.norm(accelerations, axis=1)):.6f}")


def demo_turbo_ergodic_optimization():
    """
    Demonstrate turbo-charged ergodic trajectory optimization.
    """
    print("=== TURBO-CHARGED Ergodically Trajectory Optimization Demo ===")
    print("Target: Uniform distribution over [1100,1150] × [400,420]")
    print("Method: CMA-ES with parallel processing and vectorization")
    
    # Create turbo optimizer with DOUBLE waypoints and MORE iterations
    # Unit coordinate space: X: 0-1, Y: 0-1
    optimizer = TurboErgodicTrajectoryOptimizer(
        domain_bounds=(0.0, 1.0, 0.0, 1.0),
        n_waypoints=100,  # DOUBLED from 50 to 100
        k_max=6,
        n_trajectory_samples=2000,  # Increased for better resolution
        n_cores=10,
        verbose=True,
        bspline_degree=3,  # Default B-spline degree
        ellipse_center=(0.5, 0.5),  # Center of the unit space
        ellipse_radius=0.5,  # Unit circle radius
        ellipse_eccentricity=0.0
    )
    
    # Run multi-resolution optimization with MORE iterations
    results = optimizer.optimize_multi_resolution_extended()
    
    # Visualize
    optimizer.visualize_results()
    
    # Analyze
    optimizer.analyze_ergodicity()
    
    return optimizer


def demo_gaussian_hotspots_turbo():
    """
    Demonstrate turbo-charged ergodic optimization with Gaussian hotspot target distribution.
    """
    print("\n=== TURBO-CHARGED Ergodically Optimization with Gaussian Hotspots ===")
    
    # Create optimizer with custom target distribution and DOUBLE waypoints
    # Unit coordinate space: X: 0-1, Y: 0-1
    optimizer = TurboErgodicTrajectoryOptimizer(
        domain_bounds=(0.0, 1.0, 0.0, 1.0),
        n_waypoints=100,  # DOUBLED from 50 to 100
        k_max=6,
        n_trajectory_samples=2000,  # Increased for better resolution
        n_cores=10,
        verbose=True,
        bspline_degree=3,  # Default B-spline degree
        ellipse_center=(0.5, 0.5),  # Center of the unit space
        ellipse_radius=0.5,  # Unit circle radius
        ellipse_eccentricity=0.6
    )
    
    # Override target distribution with Gaussian hotspots
    def gaussian_hotspots(x, y):
        # Single broad Gaussian centered in the middle of the unit space
        x0, y0, sigma = 0.5, 0.5, 0.2  # Centered in unit space, adjusted sigma

        # Handle both scalar and array inputs
        if np.isscalar(x):
            result = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
            # For scalar input, simple area normalization fallback
            return result / (optimizer.Lx * optimizer.Ly)
        else:
            result = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

            # Normalize to integrate to 1 over the domain (numerical integral)
            dx = optimizer.Lx / (x.shape[1] - 1) if x.shape[1] > 1 else optimizer.Lx
            dy = optimizer.Ly / (x.shape[0] - 1) if x.shape[0] > 1 else optimizer.Ly
            integral = np.sum(result) * dx * dy

            if integral > 0:
                result /= integral
            else:
                # Fallback to uniform distribution
                result = np.ones_like(x) / (optimizer.Lx * optimizer.Ly)

            return result
    
    optimizer._target_distribution = gaussian_hotspots
    
    # Use grid-based computation for non-uniform distribution
    optimizer.mu_k = optimizer._compute_target_fourier_coefficients_grid()
    
    # Run optimization with MORE iterations
    results = optimizer.optimize_multi_resolution_extended()
    
    # Visualize
    optimizer.visualize_results()
    
    # Analyze
    optimizer.analyze_ergodicity()
    
    return optimizer


if __name__ == "__main__":
    print("=== TURBO-CHARGED Ergodically Trajectory Optimization ===")
    print("This version uses CMA-ES, vectorization, and parallel processing!")
    print("ENHANCED: 100 waypoints with extended optimization time!")
    print("UPDATED: Elliptical workspace with configurable radius and eccentricity!")
    print("NEW: Configurable B-spline degree for trajectory smoothness!")
    print("NEW COORDINATE SPACE: [0,1] × [0,1] (unit square with 0.5 radius circle)")
    
    # Demo 1: Uniform distribution with turbo optimization
    uniform_optimizer = demo_turbo_ergodic_optimization()
    
    # Demo 2: Gaussian hotspots with turbo optimization
    hotspot_optimizer = demo_gaussian_hotspots_turbo()
    
    print("\n=== TURBO Performance Features ===")
    print("1. CMA-ES optimization with parallel evaluation")
    print("2. Vectorized Fourier coefficient computation")
    print("3. Analytic target distribution computation")
    print("4. Vectorized robot penalties (no Python loops)")
    print("5. Multi-resolution optimization (coarse-to-fine)")
    print("6. Parallel processing on multiple cores")
    print("7. Precomputed arrays for maximum speed")
    print("8. Optimized B-spline trajectory generation with bounds guarantee")
    print("9. EXTENDED: 100 waypoints with 3-stage optimization (100+150+200 iterations)")
    print("10. Adaptive core allocation for high-resolution problems")
    print("11. Enhanced progress tracking for longer optimization runs")
    print("12. ELLIPSE WORKSPACE: Center, radius a, eccentricity e (b=a*sqrt(1-e^2))")
    print("13. IMPROVED B-SPLINE: Projection to ellipse to ensure feasibility")
