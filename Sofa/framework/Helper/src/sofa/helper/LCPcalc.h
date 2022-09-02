/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_LCPCALC_H
#define SOFA_HELPER_LCPCALC_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sofa/helper/system/thread/CTime.h>
#include <vector>
#include <ostream>
#include <limits>	

namespace
{

    namespace Detail
    {
        double constexpr sqrtNewtonRaphson(double x, double curr, double prev)
        {
            return curr == prev
                ? curr
                : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
        }
    }

    /*
    * Constexpr version of the square root
    * Return value:
    *	- For a finite and non-negative value of "x", returns an approximation for the square root of "x"
    *   - Otherwise, returns NaN
    */
    double constexpr constsqrt(double x)
    {
        return x >= 0 && x < std::numeric_limits<double>::infinity()
            ? Detail::sqrtNewtonRaphson(x, x, 0)
            : std::numeric_limits<double>::quiet_NaN();
    }
}

namespace sofa
{

namespace helper
{

#define EPSILON_LCP		0.00000000001	// epsilon pour tests = 0
#define EPSILON_CONV	0.001			// for GS convergence
#define MAX_BOU	50	// nombre maximal de boucles de calcul

class SOFA_HELPER_API LCP
{
    int maxConst;
    double* dfree;
    double** W;
    double* f, *f_1;
    double* d;
    double tol;
    int numItMax;
    bool useInitialF;
    double mu;
    int dim;  //=3*nbContact !!
    //unsigned int nbConst;


public:
    LCP();
    ~LCP();
    void reset(void);
    void allocate (unsigned int maxConst);
    inline double** getW(void) {return W;};
    inline double& getMu(void) { return mu;};
    inline double* getDfree(void) {return dfree;};
    inline double getTolerance(void) {return tol;};
    inline double getMaxIter(void) {return numItMax;};
    inline double* getF(void) {return f;};
    inline double* getF_1(void) {return f_1;};
    inline double* getD(void) {return d;};
    inline bool useInitialGuess(void) {return useInitialF;};
    inline unsigned int getDim(void) {return dim;};
    inline unsigned int setDim(unsigned int nbC) {dim = nbC; return 0;};
    inline unsigned int getMaxConst(void) {return maxConst;};
    inline void setNumItMax(int input_numItMax) {numItMax = input_numItMax;};
    inline void setTol(double input_tol) {tol = input_tol;};

    void setLCP(unsigned int input_dim, double *input_dfree, double **input_W, double *input_f, double &input_mu, double &input_tol, int input_numItMax);

    void solveNLCP(bool convergenceTest, std::vector<double>* residuals = nullptr, std::vector<double>* violations = nullptr);
    int it; // to get the number of iteration that is necessary for convergence
    double error; // to get the error at the end of the convergence
};




inline SOFA_HELPER_API void set3Dof(double *vector, int index, double vx, double vy, double vz)
{vector[3*index]=vx; vector[3*index+1]=vy; vector[3*index+2]=vz;}
inline SOFA_HELPER_API void add3Dof(double *vector, int index, double vx, double vy, double vz)
{vector[3*index]+=vx; vector[3*index+1]+=vy; vector[3*index+2]+=vz;}
inline SOFA_HELPER_API double normError(double f1x, double f1y, double f1z, double f2x, double f2y, double f2z)
{
    return sqrt( ((f2x-f1x)*(f2x-f1x) + (f2y-f1y)*(f2y-f1y) + (f2z-f1z)*(f2z-f1z)) /
            (f1x*f1x + f1y*f1y + f1z*f1z) ) ;
}

inline SOFA_HELPER_API double absError(double f1x, double f1y, double f1z, double f2x, double f2y, double f2z)
{return sqrt ((f2x-f1x)*(f2x-f1x) + (f2y-f1y)*(f2y-f1y) + (f2z-f1z)*(f2z-f1z));}


SOFA_HELPER_API int resoudreLCP(int, double *, double **, double *);


SOFA_HELPER_API void afficheSyst(double *q,double **M, int *base, double **mat, int dim);
SOFA_HELPER_API void afficheLCP(double *q, double **M, int dim);
SOFA_HELPER_API void afficheLCP(double *q, double **M, double *f, int dim);
SOFA_HELPER_API void resultToString(std::ostream& s, double *f, int dim);

typedef double FemClipsReal;
SOFA_HELPER_API void gaussSeidelLCP1(int dim, FemClipsReal * q, FemClipsReal ** M, FemClipsReal * res, double tol, int numItMax, double minW=0.0, double maxF=0.0, std::vector<double>* residuals = nullptr);



// inverted SymMatrix 3x3 //
class SOFA_HELPER_API LocalBlock33
{
public:
    constexpr LocalBlock33() = default;
    ~LocalBlock33() {};

    void compute(double &w11, double &w12, double &w13, double &w22, double &w23, double &w33);
    void stickState(double &dn, double &dt, double &ds, double &fn, double &ft, double &fs);
    void slipState(double &mu, double &dn, double &dt, double &ds, double &fn, double &ft, double &fs);

    // computation of a new state using a simple gauss-seidel loop // pseudo-potential
    void GS_State(double &mu, double &dn, double &dt, double &ds, double &fn, double &ft, double &fs);

    // computation of a new state using a simple gauss-seidel loop // pseudo-potential (new: dn, dt, ds already take into account current value of fn, ft and fs)
    constexpr void New_GS_State(double &mu, double &dn, double &dt, double &ds, double &fn, double &ft, double &fs)
    {

        double d[3]{};

        f_1[0] = fn; f_1[1] = ft; f_1[2] = fs;

        // evaluation of the current normal position
        d[0] = dn;
        // evaluation of the new contact force
        fn -= d[0] / w[0];

        if (fn <= 0)
        {
            fn = 0; ft = 0; fs = 0;
            // if the force was previously not null -> update the state
            if (f_1[0] > 0)
            {
                double df[3]{};
                df[0] = fn - f_1[0];  df[1] = ft - f_1[1];  df[2] = fs - f_1[2];

                dn += w[0] * df[0] + w[1] * df[1] + w[2] * df[2];
                dt += w[1] * df[0] + w[3] * df[1] + w[4] * df[2];
                ds += w[2] * df[0] + w[4] * df[1] + w[5] * df[2];
            }
            return;
        }


        // evaluation of the current tangent positions
        d[1] = w[1] * (fn - f_1[0]) + dt;
        d[2] = w[2] * (fn - f_1[0]) + ds;

        // envaluation of the new fricton forces
        ft -= 2 * d[1] / (w[3] + w[5]);
        fs -= 2 * d[2] / (w[3] + w[5]);

        double normFt = constsqrt(ft * ft + fs * fs);

        if (normFt > mu * fn)
        {
            ft *= mu * fn / normFt;
            fs *= mu * fn / normFt;
        }

        double df[3]{};
        df[0] = fn - f_1[0];  df[1] = ft - f_1[1];  df[2] = fs - f_1[2];

        dn += w[0] * df[0] + w[1] * df[1] + w[2] * df[2];
        dt += w[1] * df[0] + w[3] * df[1] + w[4] * df[2];
        ds += w[2] * df[0] + w[4] * df[1] + w[5] * df[2];



    }

    // computation of a new state using biPotential approach
    void BiPotential(double &mu, double &dn, double &dt, double &ds, double &fn, double &ft, double &fs);

    void setPreviousForce(double &fn, double &ft, double &fs) {f_1[0]=fn; f_1[1]=ft; f_1[2]=fs;}

    bool computed{false};

    double w[6]{};
    double wInv[6]{};
    double det{};
    double f_1[3]{}; // previous value of force
};

// Multigrid algorithm for contacts
SOFA_HELPER_API int nlcp_multiGrid(int dim, double *dfree, double**W, double *f, double mu, double tol, int numItMax, bool useInitialF, double** W_coarse, std::vector<int> &contact_group, unsigned int num_group,  bool verbose=false);
SOFA_HELPER_API int nlcp_multiGrid_2levels(int dim, double *dfree, double**W, double *f, double mu, double tol, int numItMax, bool useInitialF,
        std::vector< int> &contact_group, unsigned int num_group, std::vector< int> &constraint_group, std::vector<double> &constraint_group_fact, bool verbose, std::vector<double>* residuals1 = nullptr, std::vector<double>* residuals2 = nullptr);
SOFA_HELPER_API int nlcp_multiGrid_Nlevels(int dim, double *dfree, double**W, double *f, double mu, double tol, int numItMax, bool useInitialF,
        std::vector< std::vector< int> > &contact_group_hierarchy, std::vector<unsigned int> Tab_num_group, std::vector< std::vector< int> > &constraint_group_hierarchy, std::vector< std::vector< double> > &constraint_group_fact_hierarchy, bool verbose, std::vector<double> *residualsN = nullptr, std::vector<double> *residualLevels = nullptr, std::vector<double> *violations = nullptr);

// Gauss-Seidel like algorithm for contacts
SOFA_HELPER_API int nlcp_gaussseidel(int dim, double *dfree, double**W, double *f, double mu, double tol, int numItMax, bool useInitialF, bool verbose = false, double minW=0.0, double maxF=0.0, std::vector<double>* residuals = nullptr, std::vector<double>* violations = nullptr);
// Timed Gauss-Seidel like algorithm for contacts
SOFA_HELPER_API int nlcp_gaussseidelTimed(int, double *, double**, double *, double, double, int, bool, double timeout, bool verbose=false);
} // namespace helper

} // namespace sofa

#endif
