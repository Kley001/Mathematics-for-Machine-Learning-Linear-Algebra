
# coding: utf-8

# # Gram-Schmidt process
# 
# ## Instructions
# In this assignment you will write a function to perform the Gram-Schmidt procedure, which takes a list of vectors and forms an orthonormal basis from this set.
# As a corollary, the procedure allows us to determine the dimension of the space spanned by the basis vectors, which is equal to or less than the space which the vectors sit.
# 
# You'll start by completing a function for 4 basis vectors, before generalising to when an arbitrary number of vectors are given.
# 
# Again, a framework for the function has already been written.
# Look through the code, and you'll be instructed where to make changes.
# We'll do the first two rows, and you can use this as a guide to do the last two.
# 
# ### Matrices in Python
# Remember the structure for matrices in *numpy* is,
# ```python
# A[0, 0]  A[0, 1]  A[0, 2]  A[0, 3]
# A[1, 0]  A[1, 1]  A[1, 2]  A[1, 3]
# A[2, 0]  A[2, 1]  A[2, 2]  A[2, 3]
# A[3, 0]  A[3, 1]  A[3, 2]  A[3, 3]
# ```
# You can access the value of each element individually using,
# ```python
# A[n, m]
# ```
# You can also access a whole row at a time using,
# ```python
# A[n]
# ```
# 
# Building on last assignment, in this exercise you will need to select whole columns at a time.
# This can be done with,
# ```python
# A[:, m]
# ```
# which will select the m'th column (starting at zero).
# 
# In this exercise, you will need to take the dot product between vectors. This can be done using the @ operator.
# To dot product vectors u and v, use the code,
# ```python
# u @ v
# ```
# 
# All the code you should complete will be at the same level of indentation as the instruction comment.
# 
# ### How to submit
# Edit the code in the cell below to complete the assignment.
# Once you are finished and happy with it, press the *Submit Assignment* button at the top of this notebook.
# 
# Please don't change any of the function names, as these will be checked by the grading script.
# 
# If you have further questions about submissions or programming assignments, here is a [list](https://www.coursera.org/learn/linear-algebra-machine-learning/discussions/weeks/1/threads/jB4klkn5EeibtBIQyzFmQg) of Q&A. You can also raise an issue on the discussion forum. Good luck!

# In[1]:


# GRADED FUNCTION
import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14

def gsBasis4(A) :
    B = np.array(A, dtype=np.float_)
    B[:, 0] = B[:, 0] / la.norm(B[:, 0])
    B[:, 1] = B[:, 1] - B[:, 1] @ B[:, 0] * B[:, 0]
    if la.norm(B[:, 1]) > verySmallNumber :
        B[:, 1] = B[:, 1] / la.norm(B[:, 1])
    else :
        B[:, 1] = np.zeros_like(B[:, 1])

    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 0] * B[:, 0]
    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 1] * B[:, 1]   

    if la.norm(B[:, 2]) > verySmallNumber :
        B[:, 2] = B[:, 2] / la.norm(B[:, 2])
    else :
        B[:, 2] = np.zeros_like(B[:, 2])    
        

    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 0] * B[:, 0]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 1] * B[:, 1]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 2] * B[:, 2]    
    
    if la.norm(B[:, 3]) > verySmallNumber :
        B[:, 3] = B[:, 3] / la.norm(B[:, 3])
    else :
        B[:, 3] = np.zeros_like(B[:, 3])        
    return B


def gsBasis(A) :
    B = np.array(A, dtype=np.float_) 
    for i in range(B.shape[1]) :
        for j in range(i) :
            B[:, i] = B[:, i] - B[:, i] @ B[:, j] * B[:, j] 
        if la.norm(B[:, i]) > verySmallNumber :
            B[:, i] = B[:, i] / la.norm(B[:, i])
        else :
            B[:, i] = np.zeros_like(B[:, i])
    return B


def dimensions(A) :
    return np.sum(la.norm(gsBasis(A), axis=0))


# ## Test your code before submission
# To test the code you've written above, run the cell (select the cell above, then press the play button [ ▶| ] or press shift-enter).
# You can then use the code below to test out your function.
# You don't need to submit this cell; you can edit and run it as much as you like.
# 
# Try out your code on tricky test cases!

# In[2]:


V = np.array([[1,0,2,6],
              [0,1,8,2],
              [2,8,3,1],
              [1,-6,2,3]], dtype=np.float_)
gsBasis4(V)


# In[3]:


# Once you've done Gram-Schmidt once,
# doing it again should give you the same result. Test this:
U = gsBasis4(V)
gsBasis4(U)


# In[4]:


# Try the general function too.
gsBasis(V)


# In[5]:


# See what happens for non-square matrices
A = np.array([[3,2,3],
              [2,5,-1],
              [2,4,8],
              [12,2,1]], dtype=np.float_)
gsBasis(A)


# In[6]:


dimensions(A)


# In[7]:


B = np.array([[6,2,1,7,5],
              [2,8,5,-4,1],
              [1,-6,3,2,8]], dtype=np.float_)
gsBasis(B)


# In[8]:


dimensions(B)


# In[9]:


# Now let's see what happens when we have one vector that is a linear combination of the others.
C = np.array([[1,0,2],
              [0,1,-3],
              [1,0,2]], dtype=np.float_)
gsBasis(C)


# In[10]:


dimensions(C)


# In[ ]:




