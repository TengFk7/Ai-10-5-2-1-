# -*- coding: utf-8 -*-
"""
Thai Coin Detector using OpenCV
================================
Detects Thai coins (1, 2, 5, 10 Baht) using:
  1. Hough Circle Transform - find circular objects
  2. Color analysis - distinguish gold vs silver
  3. Bi-metallic check - identify 10 Baht (gold ring + silver center)
  4. Relative size comparison - classify coin denominations

Works without any trained model - pure computer vision approach.
"""

import cv2
import numpy as np


class CoinDetector:
    """Detect and classify Thai coins using OpenCV."""

    # Real coin diameters (mm) for reference
    # 1 Baht: 20mm, silver (cupro-nickel)
    # 2 Baht: 21.75mm, gold/brass (aluminium bronze)
    # 5 Baht: 24mm, silver (cupro-nickel clad)
    # 10 Baht: 26mm, bi-metallic (silver center + gold ring)

    COIN_INFO = {
        0: {'name': '1 Baht',  'value': 1,  'color': [192, 192, 192]},
        1: {'name': '2 Baht',  'value': 2,  'color': [0, 215, 255]},
        2: {'name': '5 Baht',  'value': 5,  'color': [192, 192, 192]},
        3: {'name': '10 Baht', 'value': 10, 'color': [0, 165, 255]},
    }

    def __init__(self, min_radius=25, max_radius=200):
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect(self, image):
        """
        Detect coins in an image.

        Args:
            image: BGR numpy array from OpenCV

        Returns:
            list of detection dicts with keys:
              class_id, class_name, value, confidence, bbox, color
        """
        if image is None:
            return []

        h, w = image.shape[:2]

        # Adaptive radius based on image size
        img_diag = int(np.sqrt(w**2 + h**2))
        min_r = max(self.min_radius, img_diag // 60)
        max_r = min(self.max_radius, img_diag // 6)
        min_dist = max(min_r * 2, 40)

        # Pre-process
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Multi-pass circle detection with different sensitivities
        all_circles = []

        for param2 in [45, 35, 55]:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=min_dist,
                param1=100,
                param2=param2,
                minRadius=min_r,
                maxRadius=max_r,
            )
            if circles is not None:
                for c in np.round(circles[0]).astype(int):
                    all_circles.append(c)

        if not all_circles:
            return []

        # Remove duplicate circles (keep strongest)
        circles = self._deduplicate_circles(all_circles, min_dist=min_dist * 0.7)

        if not circles:
            return []

        # Validate each circle is actually a coin-like object
        valid_circles = []
        for (cx, cy, r) in circles:
            # Skip circles too close to edges
            if cx - r < 0 or cy - r < 0 or cx + r >= w or cy + r >= h:
                continue

            # Check circularity & contrast against background
            score = self._coin_likelihood(image, gray, cx, cy, r)
            if score > 0.3:
                valid_circles.append((cx, cy, r, score))

        if not valid_circles:
            return []

        # Get radii for relative sizing
        radii = [c[2] for c in valid_circles]
        max_radius_found = max(radii)
        min_radius_found = min(radii)
        radius_range = max_radius_found - min_radius_found

        detections = []
        for (cx, cy, r, coin_score) in valid_circles:
            # Analyze color properties
            color_info = self._analyze_color(image, cx, cy, r)

            # Classify
            class_id, confidence = self._classify_coin(
                r, max_radius_found, min_radius_found,
                radius_range, color_info, coin_score
            )

            info = self.COIN_INFO[class_id]

            # Bounding box
            x1 = max(0, cx - r)
            y1 = max(0, cy - r)
            x2 = min(w, cx + r)
            y2 = min(h, cy + r)

            detections.append({
                'class_id': class_id,
                'class_name': info['name'],
                'value': info['value'],
                'confidence': round(confidence, 2),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'color': info['color'],
            })

        return detections

    def _deduplicate_circles(self, circles, min_dist=50):
        """Remove overlapping circle detections, keeping best ones."""
        if not circles:
            return []

        circles = sorted(circles, key=lambda c: c[2], reverse=True)
        kept = []

        for c in circles:
            is_dup = False
            for k in kept:
                dist = np.sqrt((c[0] - k[0])**2 + (c[1] - k[1])**2)
                if dist < min_dist:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(c)

        return kept

    def _coin_likelihood(self, image, gray, cx, cy, r):
        """
        Score how likely a detected circle is actually a coin.
        Checks edge strength, circularity, and contrast.
        """
        h, w = gray.shape
        score = 0.0

        # 1. Edge strength around the perimeter
        mask_ring = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_ring, (cx, cy), r, 255, 3)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask_ring))
        ring_pixels = cv2.countNonZero(mask_ring)
        edge_ratio = edge_pixels / max(ring_pixels, 1)

        if edge_ratio > 0.15:
            score += 0.4
        elif edge_ratio > 0.08:
            score += 0.2

        # 2. Contrast between coin interior and surrounding
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(inner_mask, (cx, cy), max(r - 5, 1), 255, -1)

        outer_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(outer_mask, (cx, cy), r + 15, 255, -1)
        cv2.circle(outer_mask, (cx, cy), r + 3, 0, -1)

        inner_mean = cv2.mean(gray, mask=inner_mask)[0]
        outer_mean = cv2.mean(image, mask=outer_mask)[0] if cv2.countNonZero(outer_mask) > 0 else inner_mean

        contrast = abs(inner_mean - outer_mean)
        if contrast > 20:
            score += 0.4
        elif contrast > 10:
            score += 0.2

        # 3. Color uniformity inside the coin (coins are relatively uniform)
        coin_region = cv2.bitwise_and(image, image, mask=inner_mask)
        if cv2.countNonZero(inner_mask) > 0:
            hsv = cv2.cvtColor(coin_region, cv2.COLOR_BGR2HSV)
            h_std = np.std(hsv[:, :, 0][inner_mask > 0])
            s_std = np.std(hsv[:, :, 1][inner_mask > 0])

            if h_std < 30 and s_std < 50:
                score += 0.3
            elif h_std < 50:
                score += 0.15

        return min(score, 1.0)

    def _analyze_color(self, image, cx, cy, r):
        """
        Analyze color properties of a detected coin.

        Returns dict with:
          - is_golden: True if coin appears gold/brass
          - is_silver: True if coin appears silver/nickel
          - is_bimetallic: True if inner/outer colors differ significantly
          - mean_bgr: average BGR color
          - inner_bgr: inner region BGR
          - outer_bgr: outer ring BGR
        """
        h, w = image.shape[:2]

        # Inner circle mask (55% of radius = center of coin)
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        inner_r = max(int(r * 0.5), 3)
        cv2.circle(inner_mask, (cx, cy), inner_r, 255, -1)

        # Outer ring mask
        outer_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(outer_mask, (cx, cy), r - 2, 255, -1)
        cv2.circle(outer_mask, (cx, cy), int(r * 0.6), 0, -1)

        # Full coin mask
        full_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(full_mask, (cx, cy), r - 2, 255, -1)

        inner_bgr = np.array(cv2.mean(image, mask=inner_mask)[:3])
        outer_bgr = np.array(cv2.mean(image, mask=outer_mask)[:3])
        mean_bgr = np.array(cv2.mean(image, mask=full_mask)[:3])

        # Convert to HSV for better color analysis
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        inner_hsv = np.array(cv2.mean(hsv_image, mask=inner_mask)[:3])
        outer_hsv = np.array(cv2.mean(hsv_image, mask=outer_mask)[:3])
        mean_hsv = np.array(cv2.mean(hsv_image, mask=full_mask)[:3])

        # Golden detection: Hue in yellow-orange range (15-35), decent saturation
        is_golden = (
            (10 < mean_hsv[0] < 40) and
            (mean_hsv[1] > 40) and
            (mean_hsv[2] > 80)
        )

        # Silver: low saturation
        is_silver = mean_hsv[1] < 50 and mean_hsv[2] > 60

        # Bi-metallic: significant color difference between inner and outer
        color_diff_bgr = np.linalg.norm(inner_bgr - outer_bgr)
        hue_diff = abs(float(inner_hsv[0]) - float(outer_hsv[0]))
        sat_diff = abs(float(inner_hsv[1]) - float(outer_hsv[1]))

        is_bimetallic = (color_diff_bgr > 25) or (hue_diff > 8 and sat_diff > 15)

        return {
            'is_golden': bool(is_golden),
            'is_silver': bool(is_silver),
            'is_bimetallic': bool(is_bimetallic),
            'mean_bgr': mean_bgr,
            'inner_bgr': inner_bgr,
            'outer_bgr': outer_bgr,
            'mean_hsv': mean_hsv,
            'inner_hsv': inner_hsv,
            'outer_hsv': outer_hsv,
        }

    def _classify_coin(self, radius, max_r, min_r, r_range, color_info, coin_score):
        """
        Classify a coin based on relative size and color properties.

        Returns (class_id, confidence)
        """
        # Relative size (0-1 range)
        if r_range > 0:
            rel_size = (radius - min_r) / r_range
        else:
            rel_size = 0.5  # only one size detected

        is_golden = color_info['is_golden']
        is_silver = color_info['is_silver']
        is_bimetallic = color_info['is_bimetallic']

        # Scoring for each class
        scores = {0: 0, 1: 0, 2: 0, 3: 0}

        # ---- 10 Baht: Largest + Bi-metallic ----
        if is_bimetallic:
            scores[3] += 0.45
        if rel_size > 0.8:
            scores[3] += 0.35
        elif rel_size > 0.6:
            scores[3] += 0.15

        # ---- 5 Baht: Large + Silver ----
        if is_silver and not is_bimetallic:
            scores[2] += 0.3
        if 0.5 < rel_size <= 0.85:
            scores[2] += 0.35
        elif rel_size > 0.85:
            scores[2] += 0.2

        # ---- 2 Baht: Medium + Golden ----
        if is_golden and not is_bimetallic:
            scores[1] += 0.4
        if 0.3 < rel_size <= 0.7:
            scores[1] += 0.3
        elif rel_size <= 0.3:
            scores[1] += 0.15

        # ---- 1 Baht: Smallest + Silver ----
        if is_silver and not is_bimetallic:
            scores[0] += 0.3
        if rel_size <= 0.35:
            scores[0] += 0.4
        elif rel_size <= 0.5:
            scores[0] += 0.2

        # Pick highest scoring class
        best_class = max(scores, key=scores.get)
        best_score = scores[best_class]

        # Calculate confidence
        total = sum(scores.values())
        if total > 0:
            confidence = (best_score / total) * 0.6 + coin_score * 0.4
        else:
            confidence = 0.5

        confidence = max(0.5, min(0.98, confidence))

        return best_class, confidence
