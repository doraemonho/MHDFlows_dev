struct MVars{Aphys, Atrans, usr_var} <: MHDVars
    "x-component of velocity"
        ux :: Aphys
    "y-component of velocity"
        uy :: Aphys
    "z-component of velocity"
        uz :: Aphys
    "x-component of B-field"
        bx :: Aphys
    "y-component of B-field"
        by :: Aphys
    "z-component of B-field"
        bz :: Aphys

    # Temperatory Cache 
    "Non-linear term 1"
     nonlin1 :: Aphys
    "Fourier transform of Non-linear term"
    nonlinh1 :: Atrans

    # User Defined Vars
    "User Defined Vars"
    usr_vars :: usr_var
end

struct HMVars{Aphys, Atrans, usr_var} <: MHDVars
    "x-component of velocity"
        ux :: Aphys
    "y-component of velocity"
        uy :: Aphys
    "z-component of velocity"
        uz :: Aphys
    "x-component of B-field"
        bx :: Aphys
    "y-component of B-field"
        by :: Aphys
    "z-component of B-field"
        bz :: Aphys   

    "x-component of fourier ∇⨯B field"
       jxh :: Atrans
    "y-component of fourier ∇⨯B field"
       jyh :: Atrans
    "z-component of fourier ∇⨯B field"
       jzh :: Atrans

    "fourier Φ term to convere the coulomb guage"
        Φh :: Atrans

    # Temperatory Cache 
    "Non-linear term 1"
     nonlin1 :: Aphys
    "Fourier transform of Non-linear term"
    nonlinh1 :: Atrans

    # User Defined Vars
    "User Defined Vars"
    usr_vars :: usr_var
end

struct EMVars{Aphys, Atrans, usr_var} <: MHDVars
    "x-component of B-field"
        bx :: Aphys
    "y-component of B-field"
        by :: Aphys
    "z-component of B-field"
        bz :: Aphys
    "x-component of Curl B-field"
        ∇XB₁ :: Aphys
    "y-component of Curl B-field"
        ∇XB₂ :: Aphys
    "z-component of Curl B-field"
        ∇XB₃ :: Aphys
    "x-component of (∇XB)XB"
        ∇XBXB₁ :: Aphys
    "y-component of (∇XB)XB"
        ∇XBXB₂ :: Aphys
    "z-component of (∇XB)XB"
        ∇XBXB₃ :: Aphys

    "x-component of fourier Curl B-field"
        ∇XB₁h :: Atrans
    "y-component of fourier Curl B-field"
        ∇XB₂h :: Atrans
    "z-component of fourier Curl B-field"
        ∇XB₃h :: Atrans    
    "x-component of fourier (∇XB)XB"
        ∇XBXB₁h :: Atrans
    "y-component of fourier (∇XB)XB"
        ∇XBXB₂h :: Atrans
    "z-component of fourier (∇XB)XB"
        ∇XBXB₃h :: Atrans

    # Temperatory Cache 
    "Non-linear term 1"
     nonlin1 :: Aphys
    "Fourier transform of Non-linear term"
    nonlinh1 :: Atrans

    # User Defined Vars
    "User Defined Vars"
    usr_vars :: usr_var
end

struct CMVars{Aphys, Atrans, usr_var} <: MHDVars
    "density "
        ρ  :: Aphys
    "x-component of velocity"
        ux :: Aphys
    "y-component of velocity"
        uy :: Aphys
    "z-component of velocity"
        uz :: Aphys
    "x-component of B-field"
        bx :: Aphys
    "y-component of B-field"
        by :: Aphys
    "z-component of B-field"
        bz :: Aphys

    "x-component of fourier velocity"
       uxh :: Atrans
    "y-component of fourier velocity"
       uyh :: Atrans
    "z-component of fourier velocity"
       uzh :: Atrans

    # Temperatory Cache 
    "Non-linear term 1"
      nonlin1 :: Aphys
    "Fourier transform of Non-linear term"
     nonlinh1 :: Atrans

    "Non-linear term 2"
     nonlin2 :: Aphys
    "Fourier transform of Non-linear term"
     nonlinh2 :: Atrans

    # User Defined Vars
    "User Defined Vars"
    usr_vars :: usr_var
end