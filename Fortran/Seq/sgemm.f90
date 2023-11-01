 subroutine  gemm ( m ,  n ,  k ,  alpha ,  a ,  lda ,  b ,  ldb ,  beta
,  c ,  ldc ) 
   implicit   none 
   integer ,  intent ( in ) ::  m ,  n ,  k ,  lda ,  ldb ,  ldc 
   real ( kind = 8 ),  intent ( in ) ::  alpha ,  beta 
   real ( kind = 8 ),  intent ( in ) ::  a ( lda ,  k ),  b ( ldb ,  n ) 
   real ( kind = 8 ),  intent ( inout ) ::  c ( ldc ,  n ) 
   integer ::  i ,  j ,  l 
   real ( kind = 8 ) ::  temp 
   !$ omp parallel  do  private ( i ,  j ,  l ,  temp )  schedule (
   !static ) 
   do   j  = 1 ,  n 
     do   i  = 1 ,  m 
       temp = 0.0 
        do   l  = 1 ,  k 
          temp = temp + a ( i ,  l ) * b ( l ,  j ) 
        end   do 
       c ( i ,  j ) = alpha * temp + beta * c ( i ,  j ) 
     end   do 
   end   do 
 end
