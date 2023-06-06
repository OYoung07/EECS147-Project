# EECS147-Project
Simulation of planets using CUDA. 

Link to Project Report: https://docs.google.com/document/d/1hG70vn_KdoyivD15H0c3MkClLEw0a0TxA4wD672M6Sw/edit

Link to Youtube Presentation: 

Primary Functions:
- F_g = (G * M_1 * M_2) / d^2
- F = m * a
- v__t = V_i * a * t

Gravitational constant:
G = 6.67 x 10^-11

Body Factors:
 - Mass [1 x 1]
 - Position [1 x 3]
 - Velocity [1 x 3]
 - Radius [1 x 1]

Inelastic collision:
(m_1 * v_1) + (m_2 * v_2) = (m_1 + m_2) * v_t

Collisions:

[ B_1 ]       [ B_2 ]
*If any part of Body 1 overlaps Body 2, or vice versa, then B_1 + B_2

__Notes:__ 
 - For each Body:
    - launch kernel w/ list of all other bodies
    - calculate each iteration in seperate thread
    - reduce all force vectors
    - apply force and determine acceleration, change in velocity, and change in position
