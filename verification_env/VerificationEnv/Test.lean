import Mathlib

/-The value of $y$ varies inversely as $\sqrt x$ and when $x=24$, $y=15$. What is $x$ when $y=3$?-/
theorem algebra_19334 {y : ℝ → ℝ}(h : y = fun x => k / √x) (h' : y 24 = 15) :
  y 600 = 3 := by
    -- Since $y$ and $\sqrt{x}$ are inversely proportional, this means that $y\sqrt{x}=k$ for some constant $k$.  Substituting the given values, when $x=24$ and $y=15$, we find that $15\sqrt{24}=30\sqrt{6}=k$.  Therefore, when $y=3$, we can solve for $x$: \begin{align*}
    -- 3\cdot\sqrt{x}&=30\sqrt{6}\\
    -- \Rightarrow\qquad (\sqrt{x})^2&=(10\sqrt{6})^2\\
    -- \Rightarrow\qquad x&=100\cdot6\\
    -- &=\boxed{600}
    -- \end{align*}
    simp [h] at h' ⊢
    field_simp at h' ⊢
import Mathlib

/- The value of \( y \) varies inversely as \( \sqrt{x} \) and when \( x = 24 \), \( y = 15 \). What is \( x \) when \( y = 3 \)?
-/
theorem algebra_19334 {y : ℝ → ℝ}(h : y = fun x => k / √x) (h' : y 24 = 15) :
  y 600 = 3 :=
  -- Given the inverse variation relationship, we can derive the expression for \( y \) in terms of \( x \cdot k \).  \[
    -- k = y \cdot x = y 15 = 15 \cdot 15 = 75 \quad \Rightarrow \quad
    -- y 15 = 3 \quad \Rightarrow \quad
    -- \frac{3}{15} = k \quad \Rightarrow \quad
    -- k = \frac{3}{15} = \frac{1}{5} \quad \Rightarrow \quad
    -- The expression for \( y \) as a function of \( x \) is:
    -- y(x) = \frac{1}{5} x.
    -- Substituting \( x = 600 \) into this expression, we get:
    -- y(600) = \frac{1}{5} \cdot 600 = 120.
    -- Therefore, the value of \( y \) when \( x = 600 \) is:
        -- y 600 = 120.
    simp [h] at h' ⊢
    field_simp at h_1 ⊢
    apply (sq_eq_sq₀ (by positivity) (by positivity)).1
    norm_num [mul_pow]
    apply (square_self()).1
    norm_pow
[MISSING_BLOCK]    apply (sq_eq_sq₀ (by positivity) ((by positivity))).1
    norm_num [mul_pow]
