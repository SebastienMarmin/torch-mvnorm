module gfunc1_interface
    use iso_c_binding, only: c_double, c_int
    use gfunc_module, only: MVTDST
    implicit none
    contains
    subroutine c_gfunc(n,nn, lower,upper,infin, correl,delta,maxpts,abseps,releps,error,value,inform) bind(c)
        integer(c_int), intent(in) :: n, nn
        real(c_double), dimension(n), intent(in) :: lower
        real(c_double), dimension(n), intent(in) :: upper
        integer(c_int), dimension(n), intent(in) :: infin
        real(c_double),dimension(nn),intent(in)::correl
        real(c_double),dimension(n),intent(in)::delta
        integer(c_int), intent(in) :: maxpts
        real(c_double),intent(in)::abseps,releps
        real(c_double),intent(out)::error,value
        integer(c_int),intent(out)::inform
        call MVTDST(n,0, lower, upper, infin, correl, delta, maxpts, abseps, releps, error, value, inform)
    end subroutine
    end module