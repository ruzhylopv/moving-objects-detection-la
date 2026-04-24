# Moving Object Detection Using Background Separation

<img width="1022" height="334" alt="image" src="https://github.com/user-attachments/assets/d67c47d3-0692-4438-aae6-ac8e91b6b09e" />


## Overview

This project implements **moving object detection in video sequences** through **background separation**, using the framework of **Robust Principal Component Analysis (RPCA)**. The core idea is to decompose a data matrix into:

\[
A = L + S
\]

where:

- **A** — the original data matrix (video frames arranged as columns),
- **L** — the low-rank component representing the static background,
- **S** — the sparse component representing moving objects / foreground activity.

To solve this optimization problem, the project uses the **Inexact Augmented Lagrange Multiplier (IALM)** algorithm.

---

## Mathematical Formulation

The decomposition is formulated as the optimization problem:

\[
\min_{L,S} \; \|L\|_* + \lambda \|S\|_1
\]

subject to:

\[
A = L + S
\]

Where:

- \(\|L\|_*\) is the **nuclear norm**, encouraging low rank,
- \(\|S\|_1\) is the **L1 norm**, encouraging sparsity,
- \(\lambda\) balances the two objectives.

This convex optimization problem is solved using **IALM**.

---

## Methodology

### 1. Video-to-Matrix Conversion

Each video frame is vectorized and stacked as a column in matrix **A**.

If each frame has \(m \times n\) pixels and there are \(t\) frames:

\[
A \in \mathbb{R}^{(mn) \times t}
\]

---

### 2. RPCA via IALM

The IALM algorithm iteratively updates:

- the low-rank estimate using **Singular Value Thresholding (SVT)**,
- the sparse estimate using **soft-thresholding**,
- the Lagrange multipliers until convergence.

---

### 3. Reconstruction

After decomposition:

- **L** is reshaped into clean background frames,
- **S** is reshaped into foreground masks highlighting moving objects.

---

## Results

The method successfully separates:

- **stationary scene content** into the low-rank matrix,
- **moving entities** into the sparse matrix.

This enables accurate motion detection even in cluttered or noisy video sequences.

---

## Conclusion

This project showcases the power of matrix decomposition for separating structured and anomalous information in visual data. By solving the RPCA problem with IALM, moving object detection becomes both mathematically elegant and practically effective.

## Workshop Videos


<a href="https://www.youtube.com/watch?v=FiiLMBhPKns">Workshop video by Ostap Mnykh</a> <br/>
<a href="https://www.youtube.com/watch?v=-Qvdety7USw">Workshop video by Pavlo Ruzhylo</a> <br/>
<a href="https://youtu.be/6Rg5T1rLWm8">Workshop video by Taras Kopach</a> <br/>
