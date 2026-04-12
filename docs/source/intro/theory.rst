Theory
======

1. Mechanical models
--------------------

1.1 Mechanical problem
^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\Omega_t \subset R^3` be the spatial domain at time :math:`t`.
The mechanical problem in the quasi-static regime consists in finding the velocity field :math:`\dot u(t,x)` and the stress field :math:`\sigma(t,x)` solutions of:

        :math:`\mathbf{\sigma} + \rho\mathbf{g} = 0 \in \mathbf{\Omega}_t`,

        :math:`\frac{D\mathbf{\sigma}}{Dt}= \mathcal{M}(\mathbf{\sigma} (t),\mathbf{d},\dots )`  \in :math:`\mathbf{\Omega}_t`,

        :math:`\dot u = \dot u_0 \text{ on }  \Gamma_{u,t} ,` and :math:`\mathbf{\sigma}\cdot n = F_0 \text{ on } \Gamma_{\sigma t}`.


.. image:: figures/patata.jpg

In this set of equations, :math:`\rho` is the density, :math:`g` the vector of gravity acceleration, :math:`\mathbf{d}=\frac{1}{2}(\nabla \dot u + \nabla \dot u^T)`  the strain rate tensor, also written as :math:`\mathbf{\dot \epsilon}(t), \Gamma_{u,t}` and :math:`\Gamma_{\sigma , t }` are parts of the boundary with a given velocity :math:`(\dot u_0)` and the traction vector (:math:`F_0`), and :math:`\frac{D\sigma}{Dt}` is an objective time derivative (Jaumann rate) of :math:`\sigma` defined as:

         :math:`\frac{D\sigma}{Dt}=\dot \sigma - \omega \sigma + \sigma \omega`,  with :math:`\omega = (\nabla v − \nabla v^T)` the corotational rate tensor.

 :math:`\mathcal{M}` represents a functional constitutive law that corresponds to an elastic, elasto-plastic or elasto-visco-plastic rheology defined as:

         :math:`\mathcal{M}(\mathbf{\sigma}, \mathbf{d}, \mathbf{d_p}) = 2G ( \mathbf{d}-\mathbf{d_p}) + \lambda  \text{tr}( \mathbf{d}-\mathbf{d_p}) \mathbf{I} -\frac{G}{\eta} \mathbf{dev \sigma}`,

where :math:`{\mathbf{d_p}}` is the plastic part of the strain rate tensor :math:`\mathbf{d}`, :math:`\mathbf{I}` is the identity tensor, :math:`tr` the trace operator, :math:`G` and :math:`\lambda` are the Lamé parameters and :math:`\eta` is the viscosity.

1.2 Elastic, viscous and plastic rheologies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Various elasto-visco-plastic rheologies can be accounted for numerically. The Drucker-Prager shear failure criterion and the tensile failure criterion are commonly used to define plastic yielding, and are defined with a friction angle `φ`, a cohesion `C`, and a tensile strength :math:`T` leading to the following yield envelopes:

  :math:`{f}_{DP} =J(\mathbf{s}) + \alpha {I}_{1} - {p}_0`  and      :math:`{f}_{T} ={I}_{1} - T`,

where :math:`{I}_{1}=tr(\mathbf{\sigma})` is the first invariant of the stress tensor and :math:`J(\mathbf{s})` is the second invariant of the deviatoric stress tensor, with :math:`\mathbf{s}=\mathbf{\sigma}-p` and :math:`p` is the mean stress (or pressure). The same invariants characterize the strain tensor :math:`\mathbf{\varepsilon}`.  Conventionnally, negative stress and strain values correspond to compression and "shortening". Parameters  :math:`\alpha=\frac{6sin \phi}{3-sin \phi}` and :math:`p_0=\alpha.C.tan \phi`.


The plastic part of the strain rate tensor :math:`\mathbf{d_p}=\mathbf{d}-\mathbf{d_e}` is given by the non-associative flow rule:

   :math:`\mathbf{d_p} = \lambda_p \cdot \frac{\partial G(\sigma)}{\partial \sigma}`, and :math:`G(\sigma) = J(\sigma) + \frac{6sin\psi}{3-sin\psi} \cdot I_1(\sigma)`,

where :math:`G`  is the plastic potential, :math:`\psi` is the dilatancy angle, and :math:`\lambda_p` is the plastic multiplier.


Maxwell visco-elasticity in turn relates the deviatoric stress :math:`s` and the strain rate :math:`d`. The effective viscosity obeys a power law rheology, where the viscous deviatoric strain rate corresponds to:

      :math:`d_v = \gamma_0 \cdot J(s)^{n-1} \cdot e^{\frac{Q-LV}{RT}} \cdot s`,

where :math:`\gamma_0` is the initial fluidity, the inverse of a non-linear viscosity. :math:`T` is the absolute temperature at the onset of the model, :math:`R` is the ideal gas constant, and :math:`Q`, :math:`L` and :math:`n` are material activation energy, activation volume and power-law exponent.




1.3 Numerical implementation - ADELI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


ADELI  (Hassani et al., 1997,Chery et al., 2001) is a 3D Finite Element algorithm developed to solve the differential equations of quasi-static equilibrium and mass conservation using a time-explicit dynamic Relaxation Method (Cundall & Board, 1988). This method is well known for being able to track the onset and the development of localized elasto-plastic deformation. ADELI has been widely used to simulate a variety of tectono-volcanic and geodynamic settings (e.g.,Chery2001, Cerpa2015, Gerbault2018, RuzGinouves2021, Novoa2019, Novoa2022).

The three-dimensional space is discretized with tetrahedra, forming an unstructured mesh generated using the `GMSH <https://www.gmsh.info>`_ software (Geuzaine and Lemacle, 2009).

More details regarding the equations and method can be found in eg. Chery et al.  (2001) or Cerpa et al. (2015).

1.4 Other thermomechanical models: eg. FENICS (Felipe)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2. Geological framework (José)
------------------------------

3. Slip and Dilation Tendencies (Cécile)
----------------------------------------

Plotting slip and dilation tendencies out from a domain in which the stress tensor is available is a tool to evaluate how measured fractures and faults locally on a field area are consistent with that a priori stress field (e.g. Ritz et al, 1995).
It has been widely used in the Hazards and Geothermal communities for decades (eg. Jolie et al., 2016).

2.1 Definitions
^^^^^^^^^^^^^^^

2.2 Combined slip and dilation tendencies and other stress ratio representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

3. Krostov representations
--------------------------

4. other probabilistic analyses tools
-------------------------------------

References
----------

Cerpa, N. G., R. Araya, M. Gerbault, and R. Hassani, 2015: Relationship between slab dip and
topography segmentation in an oblique subduction zone: Insights from numerical modeling.
Geophysical Research Letters, 42 (14), 5786–5795, doi:10.1002/2015GL064047.

Chery, J., M. D. Zoback, and R. Hassani, 2001: An integrated mechanical model of the San Andreas fault in central and northern California. Journal of Geophysical Research: Solid Earth,
106 (B10), 22 051–22 066, doi:10.1029/2001jb000382

Cundall, P., and M. Board, 1988: A microcomputer program for modeling large-strain plasticity
problems. Prepared for the 6th International Congress on Numerical Methods in Geomechanics,
1988.

Gerbault, M., R. Hassani, C. Novoa Lizama, and A. Souche, 2018: Three-Dimensional Failure Pat-653
terns Around an Inflating Magmatic Chamber. Geochemistry, Geophysics, Geosystems, 19 (3),654
749–771, doi:10.1002/2017GC007174.

Geuzaine, C., and J.-F. Remacle, 2009: Gmsh: A 3-d finite element mesh generator with built-in
pre- and post-processing facilities. International Journal for Numerical Methods in Engineering,79 (11), 1309–1331, doi:https://doi.org/10.1002/nme.2579.


Hassani, R., D. Jongmands, and J. Chery, 1997: Study of plate deformation and stress in subduction processes using two-dimensional numerical modes. Journal of Geophysical Research, 102.

Novoa, C., and Coauthors, 2019: Viscoelastic relaxation : A mechanism to explain the decennial large surface displacements at the Laguna del Maule silicic volcanic complex. Earth and
Planetary Science Letters, 521, 46–59, doi:10.1016/j.epsl.2019.06.005.

Novoa, C., and Coauthors, 2022: The 2011 Cord´on Caulle eruption triggered by slip on the Liqui˜ne-Ofqui fault system. Earth and Planetary Science Letters, 583, doi:10.1016/j.epsl.2022.
117386.

Ruz Ginouves, J., M. Gerbault, J. Cembrano, P. Iturrieta, F. Saez Leiva, C. Novoa, and R. Hassani, 2021: The interplay of a fault zone and a volcanic reservoir from 3D elasto-plastic models: Rheological conditions for mutual trigger based on a field case from the Andean Southern Volcanic Zone. Journal of Volcanology and Geothermal Research, 418, doi:10.1016/j.jvolgeores.2021.107317.
