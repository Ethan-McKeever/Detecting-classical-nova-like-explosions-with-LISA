// clang -Xclang -fopenmp -I/opt/homebrew/include -L/opt/homebrew/lib -lm -lgslcblas -lgsl -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp   Nova_burst.c -o Nova_burst

#include <math.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <ctype.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sort_double.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <time.h>
#include <omp.h>

static const gsl_rng_type *rngtype;
static const gsl_rng *rng;


#define PI  3.1415926535897932	// Pi
#define year 3.15581498e7
#define biasflag 0  // flag = 0 -> use a posterior with a nova burst, 1 to construct a posterior without

double integrand(double t, double *params1, double *params2, double Tobs);
double integrate(double *params1, double *params2, double *simp3, double *simp5, double Tobs);
void pmapping(double *paramsp, double *params, double Tobs);
void pmapping_base(double *paramsp, double *params, double Tobs, int isTrueValues);
void pmapping_true(double *paramsp, double *params, double Tobs);
int massmap(double freq0, double Mc, double Mt, double *m1, double *m2, double *eta, double *q);
void Inverse(double **M, double **IM, int d);

void Fisher(double **fish6, double **fish4, int np, int npl, double *params, double *simp3, double *simp5, double Tobs, double SNR);
void FisherEvec(double **fish, double *ej, double **ev, int d);

void update(int n, int np, int npl, int nh, int ptest, int priorflag, int *who, double *heat, double *params1, double **paramsx, double *logLx, double *logpx, double **history6, double **history4, double *scale6, double *scale4, double *ej6, double *ej4, double **ev6, double **ev4, int **cnt, int **acc, double *simp3, double *simp5, double Tobs, gsl_rng *r, double *N11, double *N12, double *N21, double *N22);

int *int_vector(int N);
void free_int_vector(int *v);
double *double_vector(int N);
void free_double_vector(double *v);
double **double_matrix(int N, int M);
void free_double_matrix(double **m, int N);
double ***double_tensor(int N, int M, int L);
void free_double_tensor(double ***t, int N, int M);
int **int_matrix(int N, int M);
void free_int_matrix(int **m, int N);
double ****double_quad(int N, int M, int L, int K);
void free_double_quad(double ****t, int N, int M, int L);

#ifndef _OPENMP
#define omp ignore
#endif

gsl_rng **rvec;

int main(void)
{
    int i, j;
    int priorflag;
    int ii6, ii4, ii, jj, kk, p;
    int ptest;
    double *simp3, *simp5;
    double *paramsp;
    double *params1, **paramsx;
    double *scale6, *scale4;
    double **history6, **history4;
    double SNR, Tobs;
    double alpha, beta, gamma, *logLx, logLy, H;
    double *logpx;
    double x, y, num;
    double t_b, fdot, f_shift, freq;
    int mc, np, np_i, npl, typ;
    int nh = 10000;
    int nc = 1; // chains
    int ncc = 1;  // cold chains
    int scount, sacc, hold;
    int flag;
    int **cnt, **acc;
    double **fish6, **fish4, **cov6, **cov4;
    double *ej6, **ev6, *ej4, **ev4;
    double *heat;
    int *who;
    double *N11, *N12, *N21, *N22;
    double N1, N2, Nfac, BF, priorratio;
    
    double *pmap, *pml;
    double logLmax, logPmax;
    double logPinj;
    
    FILE *out_nova;
    FILE *chain_nova;
    FILE *burn_nova;
    FILE *high_nova;
    FILE *heats_nova;
    
    clock_t start, end;
    double cpu_time_used;
    
    const gsl_rng_type * P;
    gsl_rng * r;

    gsl_rng_env_setup();
    
    P = gsl_rng_default;
    r = gsl_rng_alloc (P);
    
    //##############################################
    //open MP modifications
    omp_set_num_threads(nc);
    rvec = (gsl_rng **)malloc(sizeof(gsl_rng *) * (nc));
    for(i = 0 ; i< nc; i++){
        rvec[i] = gsl_rng_alloc(P);
        gsl_rng_set(rvec[i] , i);
    }
    
    if(ncc > nc) ncc=nc; // can't have more cold chains than total chains
    
    // prior recovery test when ptest = 1. otherwise set to zero
    ptest = 0;
    
    // set up Simpson's rule stencils
    
    simp5 = (double*)malloc(sizeof(double)*(5));
    simp3 = (double*)malloc(sizeof(double)*(3));
    
    simp3[0] = 1.0/6.0;
    simp3[1] = 4.0/6.0;
    simp3[2] = 1.0/6.0;
    
    simp5[0] = 1.0/12.0;
    simp5[1] = 4.0/12.0;
    simp5[2] = 2.0/12.0;
    simp5[3] = 4.0/12.0;
    simp5[4] = 1.0/12.0;
    
    np = 6;
    np_i = 6;
    npl = 4;

    params1 = (double*)malloc(sizeof(double)*(np_i));
    paramsp = (double*)malloc(sizeof(double)*(np_i));
    logLx = double_vector(nh);
    logpx = double_vector(nh);
    paramsx = double_matrix(nc,np);
    scale6 = (double*)malloc(sizeof(double)*(np));
    scale4 = (double*)malloc(sizeof(double)*(npl));
    
    heat = double_vector(nc);
    who = int_vector(nc);

    N11 = (double*)malloc(sizeof(double));
    N12 = (double*)malloc(sizeof(double));
    N21 = (double*)malloc(sizeof(double));
    N22 = (double*)malloc(sizeof(double));

    N11[0] = 0.0;
    N12[0] = 0.0;
    N21[0] = 0.0;
    N22[0] = 0.0;

    cnt = int_matrix(nc,5);
    acc = int_matrix(nc,5);
    
    // hold MAP and ML parameters
    pmap = (double*)malloc(sizeof(double)*(np));
    pml = (double*)malloc(sizeof(double)*(np));
    
    // waveform parameters
    // A, phi0, alpha, beta, gamma, t_b
    
    SNR = 18.0;
    
    // physical parameters
    
    // A, phi0, f0, fdot, f_shift=Delta f / f, t_b

    Tobs = 4.0*year;

    fdot = -3.1e-15;
    freq = 0.0097;// + Tobs * fdot;      //use +Tobs * fdot for second two year observation time if comparing as in Fig. 7

    t_b = 0.33;
    //f_shift = -freq*5.0e-7;
    f_shift = -1.5/(Tobs);

    paramsp[0] = SNR;
    paramsp[1] = 0.0;
    paramsp[2] = freq;
    paramsp[3] = fdot;
    paramsp[4] = f_shift;
    paramsp[5] = t_b;
    
    pmapping_true(paramsp, params1, Tobs);
    
    out_nova = fopen("truth.txt","w");
    fprintf(out_nova,"%.14e %.14e %.14e %.14e %.14e %.14e\n", paramsp[0], paramsp[1], paramsp[2]*Tobs, fdot*Tobs*Tobs, f_shift*Tobs, t_b);
    fclose(out_nova);
    
    fish6 = double_matrix(np,np);
    fish4 = double_matrix(npl,npl);
    cov6 = double_matrix(np,np);
    cov4 = double_matrix(npl,npl);
    ej6 = (double*)malloc(sizeof(double)*(np));
    ev6 = double_matrix(np,np);
    ej4 = (double*)malloc(sizeof(double)*(npl));
    ev4 = double_matrix(npl,npl);
    
    Fisher(fish6, fish4, np, npl, paramsp, simp3, simp5, Tobs, SNR);   //Calculates full fisher and fisher_nb
    
    Inverse(fish6,cov6,np);
    Inverse(fish4,cov4,npl);
    
    FisherEvec(fish6, ej6, ev6, np);
    FisherEvec(fish4, ej4, ev4, npl);
    
    printf("\nFisher error estimates 6\n");
    for (i = 0; i < np; ++i) printf("%e ", sqrt(cov6[i][i]));      //Expected uncertainty in each parameter calculated from full fisher
    printf("\n");
    
    printf("\nFisher error estimates 4\n");
    for (i = 0; i < npl; ++i) printf("%e ", sqrt(cov4[i][i]));      //Expected uncertainty in each nonburst parameter calculated from fisher_nb
    printf("\n");

    // scaling of the parameter jumps
    for (i = 0; i < np; ++i){
        if(cov6[i][i] > 0) scale6[i] = sqrt(cov6[i][i]/(double)(np));
        else scale6[i] = sqrt(-cov6[i][i]/(double)(np));
    } 

    for (i = 0; i < npl; ++i){
        if(cov4[i][i] > 0) scale4[i] = sqrt(cov4[i][i]/(double)(np));
        else scale4[i] = sqrt(-cov4[i][i]/(double)(np));
    } 

    printf("\nGaussian jump sizes 6 \n");
    printf("%e %e %e %e %e %e\n\n", scale6[0], scale6[1], scale6[2], scale6[3], scale6[4], scale6[5]);

    printf("\nGaussian jump sizes 4 \n");
    printf("%e %e %e %e\n\n", scale4[0], scale4[1], scale4[2], scale4[3]);

    history6 = double_matrix(nc,np*nh);
    history4 = double_matrix(nc,npl*nh);

    high_nova = fopen("highL_nova.dat","w");

    // check Fisher jumps
    printf("checking Fisher jumps 6\n");
    for (ii = 0 ; ii < np ; ii++)
        {
            do
            {
                // 1 sigma jump in eigendirection ii
                for (i = 0 ; i < np ; i++)
                {
                    paramsx[0][i] = paramsp[i] + ej6[ii]*ev6[ii][i];
                }
                
                num = integrate(params1, paramsx[0], simp3, simp5, Tobs);
                logLy = -0.5*(params1[0]*params1[0] + paramsx[0][0]*paramsx[0][0])+num;
                
                // rescale jump if logLy not close to -0.5
                x = -logLy/0.2;
                if(x > 10.0) ej6[ii] /= 2.0;
                if(x < 10.0 && x > 1.5) ej6[ii] /= 1.2;
                //printf("x, ej num %e %e %e\n", x, ej[ii], num);
                
                //if(ej6[ii] < 1.0e-3) break;
            }while(x > 2.0);
            
            printf("%d %e\n", ii, logLy);
            
        }

    printf("checking Fisher jumps 4\n");
    for (ii = 0 ; ii < npl ; ii++)
        {
            do
            {
                // 1 sigma jump in eigendirection ii
                for (i = 0 ; i < npl ; i++)
                {
                    paramsx[0][i] = paramsp[i] + ej4[ii]*ev4[ii][i];
                }
                
                num = integrate(params1, paramsx[0], simp3, simp5, Tobs);
                logLy = -0.5*(params1[0]*params1[0] + paramsx[0][0]*paramsx[0][0])+num;
                
                // rescale jump if logLy not close to -0.5
                x = -logLy/0.5;
                if(x > 10.0) ej4[ii] /= 2.0;
                if(x < 10.0 && x > 1.5) ej4[ii] /= 1.2;
                //printf("x, ej num %e %e %e\n", x, ej[ii], num);
                
                //if(ej[ii] < 1.0e-3) break;
            }while(x > 2.0);
            
            printf("%d %e\n", ii, logLy);
            
        }


        
    // check Gaussian jumps
    printf("checking Gaussian jumps 6\n");
    for (ii = 0 ; ii < np ; ii++)
        {
            do
            {
                // 1 sigma jump in eigendirection ii
                for (i = 0 ; i < np ; i++)
                {
                    paramsx[0][i] = paramsp[i];
                }
                
                paramsx[0][ii] = paramsp[ii] +scale6[ii];
                
                num = integrate(params1, paramsx[0], simp3, simp5, Tobs);
                logLy = -0.5*(params1[0]*params1[0] + paramsx[0][0]*paramsx[0][0])+num;
                
                // rescale jump if logLy not close to -0.5
                x = -logLy/0.2;
                if(x > 1000.0) scale6[ii] /= 100.0;
                if(x > 10.0) scale6[ii] /= 2.0;
                if(x < 10.0 && x > 1.5) scale6[ii] /= 1.2;

                //if(scale[ii] < 1.0e-3) break;
            }while(x > 2.0);
            
            printf("%d %e\n", ii, logLy);
        }

    printf("checking Gaussian jumps 4\n");
    for (ii = 0 ; ii < npl ; ii++)
        {
            do
            {
                // 1 sigma jump in eigendirection ii
                for (i = 0 ; i < npl ; i++)
                {
                    paramsx[0][i] = paramsp[i];
                }
                
                paramsx[0][ii] = paramsp[ii] +scale4[ii];
                
                num = integrate(params1, paramsx[0], simp3, simp5, Tobs);
                logLy = -0.5*(params1[0]*params1[0] + paramsx[0][0]*paramsx[0][0])+num;
                
                // rescale jump if logLy not close to -0.5
                x = -logLy/0.5;
                if(x > 1000.0) scale4[ii] /= 100.0;
                if(x > 10.0) scale4[ii] /= 2.0;
                if(x < 10.0 && x > 1.5) scale4[ii] /= 1.2;

                //if(scale[ii] < 1.0e-3) break;
            }while(x > 2.0);
            
            printf("%d %e\n", ii, logLy);
        }

    
    //Setting initial values

    for (j = 0; j < nc; ++j)
    {
        for (i = 0; i < np; ++i) paramsx[j][i] = paramsp[i];
        for (i = npl; i < np; ++i) paramsx[j][i] = 0.0;
    }

    num = integrate(params1, paramsx[0], simp3, simp5, Tobs);
    logLx[0] = -0.5*(params1[0]*params1[0] + paramsp[0]*paramsp[0])+num;
    
    printf("num %e\n", num);
    printf("logL %e\n", logLx[0]);



    // initialize history
    for (j = 0; j < nc; ++j)
    {
        for (mc = 0 ; mc < nh ; mc++)
        {
            ii6 = (int)((double)(np)*gsl_rng_uniform(r));
            ii4 = (int)((double)(npl)*gsl_rng_uniform(r));
            
            x = gsl_ran_gaussian(r,1.0);
            
            for (i = 0 ; i < np ; i++)
            {
                history6[j][mc*np+i] =  paramsp[i] + x*ej6[ii6]*ev6[ii6][i];
            }
            for (i = 0 ; i < npl ; i++)
            {
                history4[j][mc*npl+i] =  paramsp[i] + x*ej4[ii4]*ev4[ii4][i];
            }
            
            //for (i = npl ; i < np ; i++)
            //{
            //    history[j][mc*np+i] =  0.0;
            //}
            
        }
    }
    

    logpx[0] = 0.0;
    
    logPinj = logLx[0]+logpx[0];
    
    for (j = 0; j < nc; ++j)
    {
        logpx[j] = logpx[0];
        logLx[j]= logLx[0];
    }
    
    printf("logpx %e\n", logpx[0]);
    
    for (j = 0; j < nc; ++j)
    {
        for (i = 0 ; i < 5 ; i++)
        {
            acc[j][i] = 0;
            cnt[j][i] = 0;
        }
    }
    
    for (j = 0; j < nc; ++j)
    {
        who[j] = j;
    }
    
    x = 1.3;
    if(nc > ncc)
    {
        x = exp(2.0*log(SNR/8)/(double)(nc-ncc));
    }
    
    printf("%f\n", x);
    
    if(x > 1.6) x = 1.6;
    
    for (j = 0; j < ncc; ++j) heat[j] = 1.0;
    for (j = ncc; j < nc; ++j) heat[j] = x*heat[j-1];
    
    chain_nova = fopen("chain.dat","w");
    burn_nova = fopen("burnin.dat","w");
    heats_nova = fopen("logLs.dat","w");
    
    kk = 0;
    
    logPmax = -1.0e60;
    logLmax = -1.0e60;

    
    scount = 1;
    sacc = 0;
    
    for (mc = 1 ; mc < 3000000 ; mc++)
    {
        
        alpha = gsl_rng_uniform(r); 
        
        if((nc > 1) && (alpha < 0.5))  // MCMC update or a PT swap
        {
            
            // chain swap
            scount++;
            
            alpha = (double)(nc-1)*gsl_rng_uniform(r);
            j = (int)(alpha);
            beta = exp((logLx[who[j]]-logLx[who[j+1]])/heat[j+1] - (logLx[who[j]]-logLx[who[j+1]])/heat[j]);
            alpha = gsl_rng_uniform(r);
            if(beta > alpha)
            {
                hold = who[j];
                who[j] = who[j+1];
                who[j+1] = hold;
                sacc++;
            }
            
        }
        else      // MCMC update
        {
            #pragma omp parallel for
            for(j=0; j < nc; j++)
            {
                update(j, np, npl, nh, ptest, priorflag, who, heat, params1, paramsx, logLx, logpx, history6, history4, scale6, scale4, ej6, ej4, ev6, ev4, cnt, acc, simp3, simp5, Tobs, rvec[j], N11, N12, N21, N22);
            }
            
        }
        
        // wait a bit before starting to check ML and MAP
        if(mc > 100*nh)
        {
            for (j = 0; j < ncc; ++j)
            {
                p = who[j]; // which chain is the cold chain
                x = logpx[p]+logLx[p];
                if(x > logPmax)
                {
                    logPmax = x;
                    for (i = 0 ; i < np ; i++) pmap[i] = paramsx[p][i];
                }
                if(logLx[p] > logLmax)
                {
                    logLmax = logLx[p];
                    for (i = 0 ; i < np ; i++) pml[i] = paramsx[p][i];
                }
            }
        }
        
        // add to history
        if(mc%10 == 0)
        {
            
            ii = kk%nh;
            
            for (j = 0; j < nc; ++j)
            {
                // history is held by temperature, not by chain
                if (paramsx[j][4] == 0) {
                    for (i = 0 ; i < npl ; i++)
                    {
                    history4[j][ii*npl+i] =  paramsx[j][i];
                    }
                }
                else {
                    for (i = 0 ; i < np ; i++)
                {
                    history6[j][ii*np+i] =  paramsx[j][i];
                }
                }
            }
            
            kk++;
        }
        
        // reset the counters near end of burn-in
        if(mc == 99*nh)
        {
            for (j = 0; j < nc; ++j)
            {
                for (i = 0 ; i < 5 ; i++)
                {
                    acc[j][i] = 0;
                    cnt[j][i] = 1;
                }
            }
        }
        
        
        if(mc%100000 == 0)
        {
            p = who[0];
            printf("%i %e %f %f %f %f\n", mc, logLx[p], (double)(sacc)/(double)(scount), (double)(acc[p][1])/(double)(cnt[p][1]), (double)(acc[p][2])/(double)(cnt[p][2]), (double)(acc[p][3])/(double)(cnt[p][3]));
        }
    
        if(mc%100 == 0)
        {
            fprintf(heats_nova,"%d ", mc/100);
            for (j = 0; j < nc; ++j)
            {
                fprintf(heats_nova,"%e ", logLx[who[j]]);
            }
            fprintf(heats_nova,"\n");
            
            // record cold chain parameters
            for (j = 0; j < ncc; ++j)
            {
                p = who[j];

                fdot = paramsx[p][3];
                f_shift = paramsx[p][4];
                t_b = paramsx[p][5];
                
                if(mc > 100*nh)
                {
                    fprintf(chain_nova, "%d %e %.14e %.14e %.14e %.14e %.14e %.14e\n", mc, logLx[p], paramsx[p][0], paramsx[p][1], paramsx[p][2]*Tobs, paramsx[p][3]*Tobs*Tobs, paramsx[p][4]*Tobs, paramsx[p][5]);
                    if(logLx[p] > -1.0) fprintf(high_nova, "%d %e %.14e %.14e %.14e %.14e %.14e %.14e\n", mc, logLx[p], paramsx[p][0], paramsx[p][1], paramsx[p][2]*Tobs, paramsx[p][3]*Tobs*Tobs, paramsx[p][4]*Tobs, paramsx[p][5]);
                }
                else
                {
                    fprintf(burn_nova, "%d %e %.14e %.14e %.14e %.14e %.14e %.14e\n", mc, logLx[p], paramsx[p][0], paramsx[p][1], paramsx[p][2]*Tobs, paramsx[p][3]*Tobs*Tobs, paramsx[p][4]*Tobs, paramsx[p][5]);
                }
            }
        }        
    }
    fclose(chain_nova);
    fclose(burn_nova);
    fclose(heats_nova);
 
    //N11 is the number of points in the nonburst model, N22 is # of points in burst model, N12 is number of jumps from noburst to burst, N21 for burst to nonburst
    N1 = N11[0] + N12[0];
    N2 = N22[0] + N21[0];
    //priorratio = 0.5 * exp(-0.5 * 0.4086 * pow(SNR,2.0));
    priorratio = 1.0;
    BF = N22[0] / (N11[0] * priorratio);
    
    printf("\n");
    printf("Initial\n");
    printf("logP %e\n", logPinj);
    for (j = 0 ; j < np ; j++) printf("%.15e ", paramsp[j]);
    printf("\n\n");
    printf("MAP\n");
    printf("logP %e\n", logPmax);
    for (j = 0 ; j < np ; j++) printf("%.15e ", pmap[j]);
    printf("\n\n");
    printf("ML\n");
    printf("logL %e\n", logLmax);
    for (j = 0 ; j < np ; j++) printf("%.15e ", pml[j]);
    printf("\n\n");
    printf("N11, N12, N21, N22, Bayes Factor:\n%e %e %e %e %e", N11[0], N12[0], N21[0], N22[0], BF);
    printf("\n\n");

    
    return 0;
}

void update(int n, int np, int npl, int nh, int ptest, int priorflag, int *who, double *heat, double *params1, double **paramsx, double *logLx, double *logpx, double **history6, double **history4, double *scale6, double *scale4, double *ej6, double *ej4, double **ev6, double **ev4, int **cnt, int **acc, double *simp3, double *simp5, double Tobs, gsl_rng *r, double *N11, double *N12, double *N21, double *N22)
{
    
    double alpha, beta, gamma, sign, exponent;
    double logLy, logpy, logpytemp, x, sqht, H, num;
    double *paramsy;
    int typ, flag;
    int i, j, ii, jj, p;
    int transflag, transflagup, transflagdown;
    double SNR, priorratio;
    double Deltafmin, Deltafmax, orderrange;

    Deltafmin = 1.e-8;
    Deltafmax = 1.e-3;
    orderrange = log10(Deltafmax) - log10(Deltafmin);
    
    double fmx, fmn;
    
    transflag = 0;
    transflagup = 0;
    transflagdown = 0;

    logpy = logpx[p];

    //printf("%e\n", paramsx[p][4]);

    // prior range on the frequency is 1 mHz around true
    fmx = params1[2]/Tobs+0.0005;
    fmn = params1[2]/Tobs-0.0005;
    
    gamma = 2.38/sqrt((double)(np));
    
    paramsy = double_vector(np);
    
    p = who[n];
    sqht = sqrt(heat[n]);

    SNR = params1[0] / sqrt(2.0);
    
    alpha = gsl_rng_uniform(r);

    if(paramsx[p][4] == 0.0) {N11[0] += 1.0;}
    else {N22[0] += 1.0;}
    
    if(alpha > 0.9) // gaussian jump centered on current location
    {
        typ = 1;
        beta = 3.0*gsl_rng_uniform(r);
        beta =  sqht*exp(-beta);
        
        for (i = 0 ; i < np ; i++)
        {
            paramsy[i] = paramsx[p][i];
        }
        
        if (paramsx[p][4] == 0.0){
            ii = (int)((double)(npl)*gsl_rng_uniform(r));
            paramsy[ii] = paramsx[p][ii] + beta*gsl_ran_gaussian(r,scale4[ii]);
        }
        else {
            ii = (int)((double)(np)*gsl_rng_uniform(r));
            paramsy[ii] = paramsx[p][ii] + beta*gsl_ran_gaussian(r,scale6[ii]);
        }

        // uniform draws for testing
        /*paramsy[0] = 1.0e4*gsl_rng_uniform(r);                            //what
         paramsy[1] = PI*(-2.0+4.0*gsl_rng_uniform(r));
         paramsy[2] = 0.1*gsl_rng_uniform(r);
         paramsy[3] = 1.31*gsl_rng_uniform(r);
         paramsy[4] = 3.0*gsl_rng_uniform(r);
         */
        
    }
    else if(alpha > 0.75) // differential evolution
    {
        typ = 2;
        
        ii = (int)((double)(nh)*gsl_rng_uniform(r));
        jj = (int)((double)(nh)*gsl_rng_uniform(r));
        
        beta = gsl_rng_uniform(r);
        
        x = 1.0;
        if(beta > 0.1) x = gamma;
        
        x = gsl_ran_gaussian(r,x);
       
        if(paramsx[p][4] == 0.0){
            for (i = 0 ; i < npl ; i++)
            {
               paramsy[i] = paramsx[p][i] + x*(history4[n][ii*(npl)+i]-history4[n][jj*(npl)+i]);
            }
            for (i = npl ; i < np ; i++)
            {
               paramsy[i] = 0.0;
            }
        }
        else{
            for (i = 0 ; i < np ; i++)
            {
               paramsy[i] = paramsx[p][i] + x*(history6[n][ii*np+i]-history6[n][jj*np+i]);
            }
        }
    }
    else if(alpha > 0.15)  // Fisher eigenvector jump
    {
        typ = 3;

        if(paramsx[p][4] == 0.0){
            //pick a direction
            ii = (int)((double)(npl)*gsl_rng_uniform(r));
        
            x = gsl_ran_gaussian(r,1.0);
            x *= sqht;
            
            for (i = 0 ; i < npl ; i++)
            {
                paramsy[i] = paramsx[p][i] + x*ej4[ii]*ev4[ii][i];
            }
            for (i = npl ; i < np ; i++)
            {
                paramsy[i] = 0.0;
            }
            
        }
        else {
            //pick a direction
            ii = (int)((double)(np)*gsl_rng_uniform(r));
        
            x = gsl_ran_gaussian(r,1.0);
            x *= sqht;
        
            for (i = 0 ; i < np ; i++)
            {
                paramsy[i] = paramsx[p][i] + x*ej6[ii]*ev6[ii][i];
            }
        }
    }
    else{    //Transdimensional Jump
        for(i = 0 ; i < npl ; i++) 
        {
            paramsy[i] = paramsx[p][i];
        }
        if(paramsx[p][4] == 0.0){

        sign = gsl_rng_uniform(r);
        exponent = orderrange * gsl_rng_uniform(r) - log10(Deltafmax);
        if (sign >= 0.5) {paramsy[4] = params1[2]/Tobs * pow(10.0,-exponent);}
        else {paramsy[4] = params1[2]/Tobs * -pow(10.0,-exponent);}

        paramsy[5] = gsl_rng_uniform(r);
        transflagup = 1;
    }
        else {
            paramsy[4] = 0.0;
            paramsy[5] = 0.0;
            transflagdown = 1;
        }
        transflag = 1;
    }
    cnt[p][typ]++;

    logLy = -1.0e60;
    
    flag = 0;
    
    // enforce prior ranges

    //paramsy[3] = paramsx[p][3];    //enable this line to measure parameter shifts for beta bias
    
    
    if(paramsy[5] < 0.0 || paramsy[5] > 1.0) flag = 1;        //placing burst in Tobs

       if (transflag == 0 && paramsy[4] != 0.0) {
        logpy = log(1.0/ fabs(paramsy[4]));
    }

    if(paramsy[0] < 0.0 || paramsy[0] > 1.0e6) flag = 1;
    if(paramsy[3] > 1.0 || paramsy[3] < -1.0) flag = 1;
    if(paramsy[1] > PI) paramsy[1] = -2.0*PI + paramsy[1];
    if(paramsy[1] < -PI) paramsy[1] = 2.0*PI + paramsy[1];
    if(paramsy[2] > fmx || paramsy[2] < fmn) flag = 1;  // prior on frequency

    if(paramsy[4] != 0) {
        if(fabs(paramsy[4]/paramsy[2]) < Deltafmin) {flag = 1;}
        if(fabs(paramsy[4]/paramsy[2]) > Deltafmax) {flag = 1;}
    }
    
    if(ptest == 1)
    {
        logLy = 0.0;
        logLx[p] = 0.0;
    }
    else
    {
        if(flag == 0)
        {
            num = integrate(params1, paramsy, simp3, simp5, Tobs);
            logLy = -0.5*(params1[0]*params1[0] + paramsy[0]*paramsy[0])+num;
        }
    }
    
    if (logLy > 0.0) logLy = -1.e60;
    if(flag == 1)  logpy = -1.0e60;
    
    H = (logLy-logLx[p])/heat[n]+logpy-logpx[p];

    //priorratio = 0.5 * exp(-0.5 * 0.4086 * pow(SNR,2.0));
    priorratio = 1.0;

    if (transflagup == 1) { H += log(priorratio);}
    if (transflagdown == 1) {H -= log(priorratio);}
    
    alpha = log(gsl_rng_uniform(r));
    
    if(H > alpha)
    {
        for (j = 0 ; j < np ; j++) paramsx[p][j] = paramsy[j];
        //if(transflag == 1) {printf("fshift: %e %e %e %e %e %e\n",paramsy[4], params1[4]/Tobs, logLy, logLx[p], H, alpha);}
        logLx[p] = logLy;
        logpx[p] = logpy;
        acc[p][typ]++;

        if (transflagup == 1) { N12[0] += 1.0;}
        if (transflagdown == 1) {N21[0] += 1.0;}
    }
    

    free_double_vector(paramsy);
}

void pmapping_true(double *paramsp, double *params, double Tobs)
{
    pmapping_base(paramsp, params, Tobs, 1);
}

void pmapping(double *paramsp, double *params, double Tobs)
{
    pmapping_base(paramsp, params, Tobs, 0);
}

void pmapping_base(double *paramsp, double *params, double Tobs, int isTrueValues)
{
        double alpha, beta, gamma, t_b, fdot, f_shift, f0;
    
        f0 = paramsp[2];
    
        fdot = paramsp[3];
        f_shift = paramsp[4];
        t_b = paramsp[5];

        alpha = f0*Tobs;
        beta = fdot*Tobs*Tobs;
        gamma = f_shift*Tobs;

        if (isTrueValues == 1)
        {
            printf("alpha beta gamma - %.10e %.10e %.10e\n", alpha, beta, gamma);
        }
        
        params[0] = paramsp[0];
        params[1] = paramsp[1];
        params[2] = alpha;
        params[3] = beta;
        params[4] = gamma;
        params[5] = t_b;
        
}

double integrate(double *params1, double *params2, double *simp3, double *simp5, double Tobs)
{
    double IG;
    double rmax;
    double t, tt, h, dh;
    double *I5, *I3;
    double s5, s3;
    int i, j;
    double err, ferr, tol, min;
    double cth, sth, cph;
    double tmin, tmax, hmax;
    
    tmin = 0.0;
    tmax = 1.0;
    
    tol = 1.0e-4;
    min = 1.0e-3;
    
    I5 = (double*)malloc(sizeof(double)*(5));
    I3 = (double*)malloc(sizeof(double)*(3));
    
    hmax = (tmax-tmin)/400.0;
   
    h = hmax/10.0;
    t = tmin;
    
    IG = 0.0;
    j = 0.0;
    
    do
    {
        
        do
        {
            
            dh = h/4.0;
            
            for (i = 0; i < 5; ++i)
            {
                tt = t + (double)(i)*dh;
                I5[i] = integrand(tt, params1, params2, Tobs);
            }
            for (i = 0; i < 3; ++i)
            {
                I3[i] = I5[2*i];
            }
            
            // 3 point Simpson
            s3 = 0.0;
            for (i = 0; i < 3; ++i)
            {
                s3 += I3[i]*simp3[i];
            }
            s3 *= h;
            // 5 point Simpson
            s5 = 0.0;
            for (i = 0; i < 5; ++i)
            {
                s5 += I5[i]*simp5[i];
            }
            s5 *= h;
            
            // absolute  error
            err = fabs(s5-s3)/(15.0);
            // fractional  error
            ferr = fabs(err/s5);
            
            if(ferr > tol && err > min) h /= 2.0;
            
            //printf("%e %e %e %e\n", s5, err, ferr, h);
            
        }while(ferr > tol && err > min);
        
        t += h;
        IG += s5;
        
        // we try a larger step for the next iteration
        // this might then have to be shrunk
        h *= 2.0;
        if(h > hmax) h = hmax;
        j++;
        
       //printf("%d %e\n", j, h);
        
        
    }while(t < tmax  && j < 500);
    
    if(j >= 500) // when the integrator needs many steps the likelihood won't be acceptable
    {
        IG = -1.0e10;
    }
    else
    {
        
        // subtract the overshoot
        
        h = t-tmax;
        t = tmax;
        
        dh = h/4.0;
        for (i = 0; i < 5; ++i)
        {
            tt = t + (double)(i)*dh;
            I5[i] = integrand(tt, params1, params2, Tobs);
        }
        // 5 point Simpson
        s5 = 0.0;
        for (i = 0; i < 5; ++i)
        {
            s5 += I5[i]*simp5[i];
        }
        s5 *= h;
        
        IG -= s5;
        
    }
    
    // printf("%d ", j);
    
    // printf("%d %e %e\n", j, IG, t);
    
    free(I3);
    free(I5);

    return IG;
    
}


double integrand(double t, double *params1, double *params, double Tobs)
{
    double ll;
    double phi1, phi2;
    double dphi;
    double *params2;
    
    params2 = (double*)malloc(sizeof(double)*(6));
    
    pmapping(params, params2, Tobs);
    
    if(t > params1[5]){
        phi1 = 2.0*PI*((params1[2])*t+0.5*(params1[3])*t*t+(params1[4])*(t-params1[5]))+(params1[1]);        
    }
    else{
        phi1 = 2.0*PI*((params1[2])*t+0.5*(params1[3])*t*t)+(params1[1]);
    }
    
    if(t > params2[5]){
        phi2 = 2.0*PI*((params2[2])*t+0.5*(params2[3])*t*t+(params2[4])*(t-params2[5]))+(params2[1]);
    }
    else{
        phi2 = 2.0*PI*((params2[2])*t+0.5*(params2[3])*t*t)+(params2[1]);
    }
    
    dphi = phi1-phi2;
    
    ll = params1[0]*params2[0]*cos(dphi);
    
    free(params2);
    
    return(ll);
    
}


void FisherEvec(double **fish, double *ej, double **ev, int d)
{
    int i, j, ec, sc;
    double x, maxc;
    double SNR;

    //d = 4;
    
    // printf("Evec\n");

    /*
    printf("\n");
    for (i = 0 ; i < d ; i++)
    {
        for (j = 0 ; j < d ; j++)
        {
            printf("%e ", fish[i][j]);
        }
        printf("\n");
    } */
    
    // phi, phi term is equal to SNR^2
    SNR = sqrt(fish[1][1]);
    

    gsl_matrix *m = gsl_matrix_alloc (d, d);

    for (i = 0 ; i < d ; i++)
    {
        for (j = 0 ; j < d ; j++)
        {
            gsl_matrix_set(m, i, j, fish[i][j]);
        }
    }
    
    gsl_vector *eval = gsl_vector_alloc (d);
    gsl_matrix *evec = gsl_matrix_alloc (d, d);
    
    gsl_eigen_symmv_workspace * w =
    gsl_eigen_symmv_alloc (d);
    
    ec = gsl_eigen_symmv (m, eval, evec, w);
    
    gsl_eigen_symmv_free (w);
    
 
    sc = gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);
    
    for (i = 0; i < d; i++)
    {
        ej[i] = gsl_vector_get (eval, i);
        
        printf("eigenvalue = %g\n", ej[i]);
        for (j = 0 ; j < d ; j++)
        {
            ev[i][j] = gsl_matrix_get(evec, j, i);
            printf("%f ", ev[i][j]);
        }
        printf("\n");
        
    }
    
    for (i = 0; i < d; i++)
    {
        // make sure no eigenvalue is too small
        if(ej[i] < 1.0) ej[i] = 1.0;
        // turn into 1-sigma jump amplitudes
        ej[i] = 1.0/sqrt(ej[i]);
        printf("jump %d = %g\n", i, ej[i]);
    }
        
        gsl_matrix_free (m);
        gsl_vector_free (eval);
        gsl_matrix_free (evec);
        
    

    
    return;
    
}

void Fisher(double **fish6, double **fish4, int np, int npl, double *params, double *simp3, double *simp5, double Tobs, double SNR)
{
    double SNRsq;
    double gamma, tau1, tau2, tau3, tau4, t_b, sig_gamma, freq;
    double v2, v3, v4;
    int i, j, k, l;
    int d, dl;
   
    //printf("Fisher ");

    d = np;
    dl = npl;
    
    for (i = 0; i < d; i++)
    {
        for (j = 0; j < d; j++) fish6[j][i] = 0.0;
    }
    
    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 4; j++) fish4[j][i] = 0.0;
    }

    printf("PARAMS\n");
    for (j = 0; j < d; j++)
    {
        printf("%.8e ", params[j]);
    }
    printf("\n");
    
    
    for (i = 0; i < d; i++)
    {
        for (j = 0; j < d; j++) fish6[i][j] = 0.0;
    }

    freq = params[2];
    gamma = params[4]*Tobs;
    t_b = params[5];
    tau1 = 1 - (t_b);
    tau2 = 1- pow((t_b), 2);
    tau3 = 1- pow((t_b), 3);
    tau4 = 1- pow((t_b), 4);
    v2 = pow(t_b, 2) - 2.0 * t_b + 1.0;
    v3 = pow(t_b, 3) - 3.0 * t_b + 2.0;
    v4 = pow(t_b, 4) - 4.0 * t_b + 3.0;

    SNRsq = SNR*SNR;
    sig_gamma = 1.0e-3 * Tobs * freq;

    fish6[0][0] = 1.0;
    fish6[1][1] = 1.0 * SNRsq;
    fish6[1][2] = PI * SNRsq;
    fish6[1][3] = (1.0/3.0)*PI * SNRsq;
    fish6[1][4] = PI * v2 * SNRsq;
    fish6[1][5] = -2.0 * PI * gamma * tau1 * SNRsq;
    fish6[2][2] = (4.0 / 3.0) * PI * PI * SNRsq;
    fish6[2][3] = (1.0 / 2.0) * PI * PI * SNRsq;
    fish6[2][4] = (2.0 / 3.0) * PI * PI * v3 * SNRsq;
    fish6[2][5] = -2.0 * PI * PI * gamma * tau2 * SNRsq;
    fish6[3][3] = (1.0 / 5.0) * PI * PI * SNRsq;
    fish6[3][4] = (1.0 / 6.0) * PI * PI * v4 * SNRsq;
    fish6[3][5] = -(2.0 / 3.0) * PI * PI * gamma * tau3 * SNRsq;
    fish6[4][4] = (4.0 / 3.0) * PI * PI * (3.0 * v2 - v3) * SNRsq - (1.0 / pow(gamma, 2));
    fish6[4][5] = -2.0 * PI * PI * gamma * v2 * SNRsq;
    fish6[5][5] = 4.0 * PI * PI * gamma * gamma * tau1 * SNRsq;

    i = 1;
    j = 1;
    while(i < d) {
        j = 1;
        while(j < i) {
            fish6[i][j] = fish6[j][i];
            j +=1;
        }
        i+=1;
    }

    fish4[0][0] = 1;
    i = 1;
    j = 1;
    while(i < dl) {
        j = 1;
        while(j < 4) {
            fish4[i][j] = fish6[i][j];
            j += 1;
        }
        i += 1;
    }

    // to avoid a singular matrix
    //if(fish6[d-1][d-1] < 1.0) fish6[d-1][d-1] = 1.0;
    
    printf("\n");
    for (i = 0; i < d; i++)
    {
        for (j = 0; j < d; j++)
        {
            printf("%.8e ", fish6[i][j]);
        }
        printf("\n");
    }
    printf("\n");

}


int *int_vector(int N)
{
    return malloc( (N+1) * sizeof(int) );
}

void free_int_vector(int *v)
{
    free(v);
}

int **int_matrix(int N, int M)
{
    int i;
    int **m = malloc( (N+1) * sizeof(int *));
    
    for(i=0; i<N+1; i++)
    {
        m[i] = malloc( (M+1) * sizeof(int));
    }
    
    return m;
}

void free_int_matrix(int **m, int N)
{
    int i;
    for(i=0; i<N+1; i++) free_int_vector(m[i]);
    free(m);
}

double *double_vector(int N)
{
    return malloc( (N+1) * sizeof(double) );
}

void free_double_vector(double *v)
{
    free(v);
}

double **double_matrix(int N, int M)
{
    int i;
    double **m = malloc( (N+1) * sizeof(double *));
    
    for(i=0; i<N+1; i++)
    {
        m[i] = malloc( (M+1) * sizeof(double));
    }
    
    return m;
}

void free_double_matrix(double **m, int N)
{
    int i;
    for(i=0; i<N+1; i++) free_double_vector(m[i]);
    free(m);
}

double ***double_tensor(int N, int M, int L)
{
    int i,j;
    
    double ***t = malloc( (N+1) * sizeof(double **));
    for(i=0; i<N+1; i++)
    {
        t[i] = malloc( (M+1) * sizeof(double *));
        for(j=0; j<M+1; j++)
        {
            t[i][j] = malloc( (L+1) * sizeof(double));
        }
    }
    
    return t;
}

void free_double_tensor(double ***t, int N, int M)
{
    int i;
    
    for(i=0; i<N+1; i++) free_double_matrix(t[i],M);
    
    free(t);
}

double ****double_quad(int N, int M, int L, int K)
{
    int i,j,k;
    
    double ****t = malloc( (N+1) * sizeof(double **));
    for(i=0; i<N+1; i++)
    {
        t[i] = malloc( (M+1) * sizeof(double *));
        for(j=0; j<M+1; j++)
        {
            t[i][j] = malloc( (L+1) * sizeof(double));
            for(k=0; k<L+1; k++)
            {
                       t[i][j][k] = malloc( (K+1) * sizeof(double));
            }
        }
    }
    
    return t;
}

void free_double_quad(double ****t, int N, int M, int L)
{
    int i;
    
    for(i=0; i<N+1; i++) free_double_tensor(t[i],M,L);
    
    free(t);
}


void Inverse(double **M, double **IM, int d)
{
    int i, j;
    int s;
    double x, maxc;
    
    gsl_matrix *m = gsl_matrix_alloc (d, d);
    
    for (i = 0 ; i < d ; i++)
    {
        for (j = 0 ; j < d ; j++)
        {
            gsl_matrix_set(m, i, j, M[i][j]);
        }
    }
    
    gsl_permutation *p = gsl_permutation_alloc(d);
    
    // Compute the LU decomposition of this matrix
    gsl_linalg_LU_decomp(m, p, &s);
    
    // Compute the  inverse of the LU decomposition
    gsl_matrix *inv = gsl_matrix_alloc(d, d);
    
    gsl_linalg_LU_invert(m, p, inv);
    
    gsl_permutation_free(p);
    
    
    for (i = 0; i < d; i++)
    {
        for (j = 0 ; j < d ; j++)
        {
            IM[i][j] = gsl_matrix_get(inv, j, i);
        }
    }
    
    printf("\n");
    for (i = 0; i < d; i++)
    {
        for (j = 0; j < d; j++)
        {
            printf("%.8e ", IM[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    if (d == 6) {
        FILE *fisher_inv_nova;
        fisher_inv_nova = fopen("fisher_inv.dat", "w");
        for (i = 0; i < d; i++)
        {
            for (j = 0; j < d; j++)
            {
                fprintf(fisher_inv_nova,"%.8e ", IM[i][j]);
            }
            fprintf(fisher_inv_nova, "\n");
        }
        fclose(fisher_inv_nova);
    }
    
    gsl_matrix_free (inv);
    gsl_matrix_free (m);
    
    return;
    
}

