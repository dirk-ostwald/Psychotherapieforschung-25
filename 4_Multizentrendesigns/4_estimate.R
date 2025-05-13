
# R code for the estimation of a linear mixed model (LMM) using the
# restricted maximum likelihood (ReML) method.  
#-- -------------------------------------------------------------------------

gls = function(y, X, V){
    # Diese Funktion bestimmt den generalisierten Kleinste-Quadrate-Schätzer. 
    # 
    # Inputs
    #   y           : y x 1 Datenvektor
    #   X           : n x p Designamtrix
    #   V           : n x n marginale Datenkovarianzmatrix
    #
    # Outputs 
    #   beta_hat    : p x 1 generalisierter Kleinste-Quadrate-Schätzer
    # ------------------------------------------------------------------------- 
    Vi          = solve(V)                                                      # Inverse 
    beta_hat    = solve(t(X) %*% Vi %*% X) %*% t(X) %*% Vi %*% y                # GKQ Schätzer
    return(beta_hat)                                                            # Output
}

llh_reml = function(theta, y, X, Z, beta_hat, Sigma_eps){
    # Diese Funktion evaluiert die negative restricted log likelihood 
    # Zielfunktion für  das Random-Effects-Modell der Metanalyse.
    #
    # Inputs
    #   theta       : k x 1 Varianzkomponentenvektor
    #   y           : n x 1 Datenvektor                                 
    #   X         : n x p Fixed-Effects-Designmatrix
    #   Z         : n x q  Random-Effects-Designmatrix
    #   beta_hat  : p x 1 Fixed-Effects Schätzer
    #
    # Outputs
    #   llh_reml    : 1 x 1 Wert der ReML Zielfunktion  
    # ------------------------------------------------------------------------- 
    n           = nrow(Z)                                                       # Datenpunktanzahl
    V           = theta[1]*Z %*% t(Z) + theta[2]*diag(n)                        # marginale Datenkovarianzmatrix
    Vi          = solve(V)                                                      # Inverse
    R           = y - X%*%beta_hat                                              # Residuals
    T1          = -(1/2)*log(det(V))                                            # Erster Term 
    T2          = -(1/2)*log(det(t(X) %*% Vi %*% X))                            # Zweiter Term
    T3          = -(1/2)*t(R) %*% Vi %*% R                                      # Dritter Term
    llh_reml    = T1 + T2 + T3                                                  # Restricted Log Likelihood
    return(llh_reml)                                                            # Wert der ReML Zielfunktion
}

reml  = function(theta, lmm){
    # Diese Funktion ist eine Wrapperfunktion für l_reml() zum Gebrauch mit 
    # der generischen R Optimierungsfunktion optim().
    #
    # Inputs
    #   theta   : k x 1 Varianzkomponentenvektor
    #   lmm     : Liste von LMM Komponenten
    #
    # Output
    #   reml    : Wert der restricted log likelhood Funktion
    # ------------------------------------------------------------------------- 
    y           = lmm$y                                                         # Datenvektor
    X           = lmm$X                                                         # Fixed-Effects-Designmatrix
    Z           = lmm$Z                                                         # Random-Effects-Designmatrix
    beta_hat    = lmm$beta_hat                                                  # Fixed-Effects Schätzer
    l_reml      = llh_reml(theta,y,X,Z,beta_hat)                                # Wert der ReML Zielfunktion
    return(-l_reml)                                                             # Ausgabeargument
}

mcov = function(theta, Z){
    # Diese Funktion schätzt generierte eine marginale Datenkovarianzmatrix
    # basierend auf einer Random-Effects-Designmatrix und der Varianzkomponenten.
    # 
    # Inputs: 
    #   theta   : c x 1 Varianzkomponentenvektor
    #   Z       : n x q Random-Effects-Designmatrix
    # 
    # Outputs
    #   V_theta : n x n marginale Kovarianzmatrix
    # -------------------------------------------------------------------------
    V_theta = theta[1]*(Z%*%t(Z)) + theta[2]*diag(nrow(Z))                      # marginale Datenkovarianzmatrix
    return(V_theta)                                                             # Ausgabeargument
}

rfx = function(lmm){
    # Diese Funktion bestimmt den bedingten Erwartungswert der Random-Effects
    #   Inputs :
    #     lmm   : R Liste mit Einträgen
    #       $y              : n x 1 Datenvektor
    #       $X              : n x p Fixed-Effects-Designmatrix
    #       $Z              : n x q Random-Effects-Designmatrix
    #       $beta_hat       : p x 1 Fixed-Effects-Parameterschätzer
    #       $s_b_hat        : 1 x 1 Random-Effects-Varianzkomponente    
    #       $s_eps_hat      : 1 x 1 Fehler-Varianzkomponente 
    #   Outputs :
    #     lmm               : R Liste mit zusätzlichen Einträgen
    #       $b_hat          : q x 1 Random-Effects-Parameterschätzer
    # ------------------------------------------------------------------- 
    y               = lmm$y                                                     # Daten
    Z               = lmm$Z                                                     # Random-Effects-Designmatrix
    X               = lmm$X                                                     # Fixed-Effects-Designmatrix
    beta_hat        = lmm$beta_hat                                              # Fixed-Effects-Parameterschätzer
    s_b_hat         = lmm$s_b_hat                                               # Random-Effects-Varianzkomponentenschätzer
    s_eps_hat       = lmm$s_eps_hat                                             # Fehlervarianzkomponentenschätzer
    theta_hat       = c(s_b_hat,s_eps_hat)                                      # Varianzkomponentenschätzer  
    V_theta_hat_i   = solve(mcov(theta_hat, Z))                                 # Inverser Datenkovarianzmatrixschätzer
    eps_hat         = (y - X %*% beta_hat)                                      # Residuals
    lmm$b_hat       = s_b_hat*t(Z) %*% V_theta_hat_i %*% eps_hat                # Random-Effects-Parameterschätzer
}

estimate = function(lmm){
    # Diese Funktion schätzt die Parameter eines LMMs.
    #   Inputs :
    #     lmm   : R Liste mit Einträgen
    #       $y              : n x 1 Datenvektor
    #       $X              : n x p Fixed-Effects-Designmatrix
    #       $Z              : n x q Random-Effects-Designmatrix
    #       $c              : 1 x 1 Varianzkomponentenanzahl
    #   Outputs :
    #     lmm   : R Liste mit zusätzlichen Einträgen
    #       $beta_hat       : p x 1 Fixed-Effects-Parameterschätzer
    #       $s_b_hat        : 1 x 1 Random-Effects-Varianzschätzer
    #       $s_eps_hat      : 1 x 1 Datenvarianzschätzer
    # -------------------------------------------------------------------------   
    y               = lmm$y                                                     # Datenvektor
    X               = lmm$X                                                     # Fixed-Effects-Designmatrix
    Z               = lmm$Z                                                     # Random-Effects-Designmatrix
    c               = lmm$c                                                     # Anzahl Varianzkomponenten
    n               = nrow(X)                                                   # Anzahl Datenpunkte
    p               = ncol(X)                                                   # Anzahl Fixed-Effects
    q               = ncol(Z)                                                   # Anzahl Random-Effects
    K               = 2^3                                                       # maximale Iterationsanzahl
    theta_hat_k     = matrix(rep(NaN, c*K), nrow = c)                           # Varianzkomponentenschätzerarray  
    theta_hat_k[,1] = rep(1,c)                                                  # Initialisierung 
    beta_hat_k      = matrix(rep(NaN, p*K), nrow = p)                           # Fixed-Effects-Schätzerarray
    V_theta_hat_k   = mcov(theta_hat_k[,1], Z)                                  # marginale Datenkovarianzmatrix
    beta_hat_k[,1]  = gls(y,X,V_theta_hat_k)                                    # Fixed-Effects-Parameterschätzer
    for (k in 2:K){                                                             # Iterationen
        lmm$beta_hat    = beta_hat_k[, k-1]                                     # Fixed-Effects-Schätzer k-1
        max_l_reml      = optim(par=theta_hat_k[,k-1],fn=reml,lmm=lmm)          # ReML-Varianzkomponentschätzung
        theta_hat_k[,k] = max_l_reml$par                                        # Varianzkomponentenschätzer k
        V_theta_hat_k   = mcov(theta_hat_k[,k], Z)                              # marginale Datenkovarianzmatrix
        beta_hat_k[,k]  = gls(y,X,V_theta_hat_k)}                               # Fixed-Effects-Parameterschätzer
    lmm$beta_hat    = beta_hat_k[,K]                                            # Fixed-Effects-Parameterschätzer
    lmm$s_b_hat     = theta_hat_k[1,K]                                          # Random-Effects-Varianzkomponente    
    lmm$s_eps_hat   = theta_hat_k[2,K]                                          # Fehler-Varianzkomponente 
    lmm$b_hat       = rfx(lmm)                                                  # Random-Effects-Parameterschätzer
    return(lmm)                                                                 # Ausgabe
}

evaluate = function(lmm){

    # Diese Funktion evaluiert ein geschätztes LMMs.
    #   Inputs:
    #       .lmm    : R Liste mit Einträgen
    #        $y          : n x 1 Datenvektor
    #        $X          : n x p Fixed-Effects-Designmatrix
    #        $Z          : n x q Random-Effects-Designmatrix
    #        $c          : 1 x 1 Varianzkomponentenanzahl
    #        $beta_hat   : p x 1 Fixed-Effects-Parameterschätzer
    #        $s_b_hat    : 1 x 1 Random-Effects-Varianzschätzer
    #        $s_eps_hat  : 1 x 1 Datenvarianzschätzer          
    #   Outputs
    #       .lmm    : R Liste mit zusätzlichen Einträgen
    #        $C_beta_hat : p x p Fixed-Effects Kovarianzmatrixschätzer
    #        $se_beta_hat: p x 1 Fixed Effects Standardfehlerschätzer     
    #        $ci_beta_hat: p x 2 Wald Fixed Effects Wald Konfidenzintervalle
    #
    # -------------------------------------------------------------------------
    y               = lmm$y                                                     # Daten
    Z               = lmm$Z                                                     # Random-Effects-Designmatrix
    X               = lmm$X                                                     # Fixed-Effects-Designmatrix
    beta_hat        = lmm$beta_hat                                              # Fixed-Effects-Parameterschätzer
    s_b_hat         = lmm$s_b_hat                                               # Random-Effects-Varianzkomponentenschätzer
    s_eps_hat       = lmm$s_eps_hat                                             # Fehlervarianzkomponentenschätzer
    theta_hat       = c(s_b_hat,s_eps_hat)                                      # Varianzkomponentenschätzer  
    V_hat_i         = solve(mcov(theta_hat, Z))                                 # inverser Datenkovarianzmatrixschätzer
    lmm$C_beta_hat  = solve(t(X) %*% V_hat_i %*% X)                             # Fixed-Effects Kovarianzmatrixschätzer    
    lmm$se_beta_hat = sqrt(diag(lmm$C_beta_hat))                                # Fixed-Effects Standardfehlerschätzer            
    lmm$gamma       = 0.95                                                      # Konfidenzlevel    
    lmm$zgamma      = qnorm((1+lmm$gamma)/2)                                    # Wald-Konfidenzintervalle (Demidenko (2013, S. 133))                            
    lmm$kappa_u     = lmm$beta_hat - lmm$zgamma*lmm$se_beta_hat                 # untere KI Grenzen
    lmm$kappa_o     = lmm$beta_hat + lmm$zgamma*lmm$se_beta_hat                 # obere KI Grenzen
    return(lmm) 
}


