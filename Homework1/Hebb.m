   f   u   n   c   t   i   o   n       [   W   ]   =   H   e   b   b   (   X   ,   d   )   
   %           $   A   u   t   h   o   r   :       A   l   e   x       G   i   b   b   o   n   s   
   %   i   n   p   u   t   :   
   %           X       =       i   n   p   u   t       v   e   c   t   o   r   
   %           d       =       d   e   s   i   r   e   d       o   u   t   p   u   t   (   s   )   
   %   o   u   t   p   u   t   :   
   %           W       =       v   e   c   t   o   r       o   f       w   e   i   g   h   t   s   
   
   
           i   f       n   a   r   g   i   n   <   2   ,   e   r   r   o   r   (   '   B   o   t   h       i   n   p   u   t       v   e   c   t   o   r       a   n   d       d   e   s   i   r   e   d       o   u   t   p   u   t   s       r   e   q   u   i   r   e   d   '   )   ,   e   n   d   
   
           W       =       z   e   r   o   s   (   1   ,   h   e   i   g   h   t   (   X   )   )   ;   
   
           f   o   r       i       =       1   :   h   e   i   g   h   t   (   X   )   
                   W   (   i   )       =       (   1   /   l   e   n   g   t   h   (   X   )   )   *   d   o   t   (   d   ,   X   (   i   ,   :   )   )   ;   
           e   n   d   
           d   i   s   p   (   "   W   e   i   g   h   t   i   n   g       v   e   c   t   o   r       f   r   o   m       i   n   p   u   t   s       a   n   d       d   e   s   i   r   e   d       o   u   t   p   u   t   s   :       "   )   ;   
           d   i   s   p   (   W   )   ;   
   e   n   d