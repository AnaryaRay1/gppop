SUBROUTINE t_cholesky_lower(n,t,l)
!*****************************************************************************80
!
!! T_CHOLESKY_LOWER: lower Cholesky factor of a Toeplitz matrix.
!
!  Discussion:
!
!    The first row of the Toeplitz matrix A is supplied.
!
!    The Toeplitz matrix must be positive semi-definite.
!
!    After factorization, A = L * L'.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    29 January 2017
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Michael Stewart,
!    Cholesky factorization of semi-definite Toeplitz matrices.
!    Linear Algebra and its Applications,
!    Volume 254, pages 497-525, 1997.
!
!  Parameters:
!
!    Input, integer n, the order of the matrix.
!
!    Input, double precision t(N), the first row.
!
!    Output, double precision l(n,n), the lower Cholesky factor.
!
      implicit none

      integer, intent(IN) :: n
      double precision, intent(IN), dimension(n) :: t
      double precision, intent(OUT), dimension(n,n) :: l

!f2py intent(in) n
!f2py intent(in) t
!f2py intent(out) l
!f2py depend(n) t
      

      double precision, dimension(2,n) :: g
      double precision, dimension(2,2) :: h
      integer i
      double precision rho
      double precision one
      one=1.0

      g(1,1:n) = t(1:n)
      g(2,1) = 0.0
      g(2,2:n) = t(2:n)

      l(1:n,1:n) = 0.0

      l(1:n,1) = g(1,1:n)
      g(1,2:n) = g(1,1:n-1)
      g(1,1) = 0.0
      do i = 2, n
        rho = - g(2,i) / g(1,i)
        h = reshape ( (/ one, rho, rho, one /), (/ 2, 2 /) )
        g(1:2,i:n) = matmul ( h, g(1:2,i:n) ) &
          / sqrt ( ( 1.0 - rho ) * ( 1.0 + rho ) )
        l(i:n,i) = g(1,i:n)
        g(1,i+1:n) = g(1,i:n-1)
        g(1,i) = 0.0
      end do

      return
end SUBROUTINE t_cholesky_lower
