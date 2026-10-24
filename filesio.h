/* 
 * File:   filesio.h
 * Author: sousasag
 *
 * Created on April 23, 2011, 2:21 PM
  *
 * Version 2 of filesio.h.  (Updated with help of opencode and deepseek free)
 *
 * The FITS reader (read_fits_file) now accepts, besides the standard
 * image with WCS keywords (CDELT1/CRVAL1/CRPIX1), spectra stored in a
 * FITS table:
 *
 *   - binary or ASCII table HDUs;
 *   - wavelength columns (in order of priority):
 *         WAVE_AIR -> WAVE -> WAVELENGTH
 *   - flux columns (in order of priority):
 *         FLUX_EL -> FLUX -> FLUX_CAL
 *   - scalar columns (one row per pixel) and vector columns with a
 *     single row (e.g. ESPRESSO: 1 row x 443262 elements);
 *   - spectra stored in image extensions, not only in the primary HDU.
 *
 * Tables with more than one row and vector columns (one spectrum per
 * row) are not supported and are rejected with an explicit message.
 *
 * The interface to the rest of ARES is unchanged: read_spectrum_file()
 * still returns pixels2 = flux vector, xpixels2 = wavelength vector,
 * npoints, cdelta1 and crval1.
*/

#ifndef _FILESIO_H
#define	_FILESIO_H

#ifdef	__cplusplus
extern "C" {
#endif

#include <string.h>
#include <ctype.h>
#include "fitsio.h"

    /* leitura das opcoes , mine.opt file read*/
//void read_mine(char * mine, char* filetest, char* fileleitura, char* fileout, double* lambdai, double* lambdaf, double* smoothder, double* tree, double* space, double* tree, double* distlinha, double* miniline, int* plots_flag);

long file_lines(char * file);
char* getFileExtension(char * name);
int set_tipo_espectro(char * extensao);

long find_pixel_line(double * xpixels, long npoints, double linha);
void read_mine(char * mine, char* filetest, char* fileleitura, char* fileout, double* lambdai, double* lambdaf, double* smoothder, double* space, char* tree, double* distlinha, double* miniline, int* plots_flag, char* rvmask);
void read_spectrum_file(char * filetest, long* npoints, double** pixels2, double** xpixels2, double* cdelta1, double* crval1, int spectrum_type);
void read_fits_file(char * filetest, long* npoints, double** pixels2, double** xpixels2, double* cdelta1, double* crval1);
void read_ascii_file(char * filetest, long* npoints, double** pixels2, double** xpixels2, double* cdelta1mean, double* crval1);
void read_lines_list(char * fileleitura, long* nl2, double** linhas);
void write_outfile(char *fileout, double* aponta, int nl, int miniline);


long find_pixel_line(double * xpixels, long npoints, double linha){
    long nctest;
    double restest[2];
    double cdelta1=xpixels[1]-xpixels[0];
    double cdelta_last=xpixels[npoints-1]-xpixels[npoints-2];
    double cdelta_mid=xpixels[npoints/2]-xpixels[npoints/2-1];
    double crval1=xpixels[0];
//    restest[0]=1./cdelta1;
    restest[0]=1./cdelta_mid;
    restest[1]=-crval1/cdelta1;
    nctest=(long) (restest[0]*linha+restest[1]);
    nctest++;
    // implementar verificação de proximidade da linha. Para o caso de cdeltas nao equidistantes. Neste caso podemos implementar um if para ver se está suficientemente perto.
    // Se nao estiver perto um while até se encontrar perto depois de verificar se tem de somar ou subtrair (cuidado com os limites)
    // Tive de mudar o cdelta para o meio do espectro pois para o ESPRESSO ficava muito longe..
    if (cdelta1 != cdelta_last || nctest < 0 || nctest > npoints -1) {
      if (nctest < 0)  nctest = 0;
      if (nctest > npoints-1)  nctest = npoints-1;
      printf("\nRefining search for central wavelenght: Inicial wavelenght: %f, index: %ld\n", xpixels[nctest], nctest);
      if (xpixels[nctest] < linha) {
        while(xpixels[nctest] < linha)
          nctest++;
      } else {
        while(xpixels[nctest] > linha)
          nctest--;
      }
    }
    return nctest;
}


void write_outfile(char *fileout, double* aponta, int nl, int miniline){
   
    
    FILE *pFile2;
    pFile2 = fopen (fileout,"wt");
    int ilinha;
    for (ilinha=0;ilinha<nl;ilinha++) {
        if (aponta[ilinha*12+4] > miniline && aponta[ilinha*12+4] < 500. && aponta[ilinha*12+1] > 0)
            fprintf(pFile2," %10.3f  %ld  %10.5f  %10.5f  %10.5f  %10.5f  %10.5f  %10.5f  %10.5f\n", aponta[ilinha*12+0], (long) aponta[ilinha*12+1], aponta[ilinha*12+2], aponta[ilinha*12+3], aponta[ilinha*12+4], aponta[ilinha*12+8], aponta[ilinha*12+5], aponta[ilinha*12+6], aponta[ilinha*12+7]);
/*  Comment the previous line, and uncomment the next line 
    if you want to output the errors for the gaussian coeficients 
    for the fitted line*/
//            fprintf(pFile2," %10.3f  %ld  %10.5f  %10.5f  %10.5f  %10.5f  %10.5f  %10.5f  %10.5f  %10.5f  %10.5f  %10.5f\n", aponta[ilinha*12+0], (long) aponta[ilinha*12+1], aponta[ilinha*12+2], aponta[ilinha*12+3], aponta[ilinha*12+4], aponta[ilinha*12+8], aponta[ilinha*12+5], aponta[ilinha*12+6], aponta[ilinha*12+7], aponta[ilinha*12+9], aponta[ilinha*12+10], aponta[ilinha*12+11]);

    }
    fclose (pFile2);
}

long file_lines(char * file) {
    FILE * pFile;
    long n = 0;
    char str[200];
    pFile = fopen (file,"rt");
    if (pFile==NULL) perror ("Error opening file");
    else
    {
    while (!feof(pFile)) {
            fgets (str , 200, pFile);
            n++;
            }
    fclose (pFile);
    }
    return (n);
}



void read_lines_list(char * fileleitura, long* nl2, double** linhas){
    long nl;
    char str[200], * pch;
    nl=file_lines(fileleitura);
    double* lab=(double *) malloc((nl-2) * sizeof(double));
    if (lab == NULL) {
            printf("Memory allocation error\n");
            exit(1);
    }

    FILE * pFile = fopen (fileleitura,"rt");
    fgets (str , 200, pFile);
    fgets (str , 200, pFile);
    int il;
    for (il=0; il < nl-3; il++) {
            fgets (str , 200, pFile);
            pch = strtok (str," ");
            char *stopstring;
            lab[il]=strtod(pch,&stopstring);
    }
    fclose (pFile);
    *linhas=lab;
    *nl2=nl-2;
}

int set_tipo_espectro(char * extensao){
    if (!strcmp(extensao, ".fits") || !strcmp(extensao, ".FITS"))
        return 0; // FITS file input
    else return 1; // ASCII file input
}


char* getFileExtension(char * name){
    printf("FILE: %s\n",name);
    char * extension= (char *) malloc(10*sizeof(char));
    char* peek = name + strlen (name) - 1;
    while (peek >= name) {
        if (*peek == '.') {
            strcpy (extension, peek);
            break;
        }
        peek--;
    }
    return extension;
}



void read_spectrum_file(char * filetest, long* npoints, double** pixels2, double** xpixels2, double* cdelta1, double* crval1, int spectrum_type){
    if (spectrum_type == 0){
        read_fits_file(filetest, npoints, pixels2, xpixels2, cdelta1, crval1);
    } else {
        read_ascii_file(filetest, npoints, pixels2, xpixels2, cdelta1, crval1);
        printf("\n ASCII READ\n");
    }

}


void read_ascii_file(char * filetest, long* npoints, double** pixels2, double** xpixels2, double* cdelta1mean, double* crval1){    

    long nl;
    char str[200], * pch, *stopstring;
    double *pixels, *xpixels;
    nl=file_lines(filetest)-1;
    

    pixels = (double *) malloc((nl-2) * sizeof(double));
    xpixels= (double *) malloc((nl-2) * sizeof(double));
    if (pixels == NULL || xpixels == NULL) {
            printf("Memory allocation error\n");
            exit(1);
    }
    *pixels2=pixels;
    *xpixels2=xpixels;
    FILE * pFile = fopen (filetest,"rt");

    double meanx=0.;
    double xtmp;
    double ftmp;

    fgets (str , 200, pFile);
    fgets (str , 200, pFile);
    long il=0;
            fgets (str , 200, pFile);
            sscanf (str,"%lf %lf",&xtmp,&ftmp);
            xpixels[il]=xtmp;
            pixels[il]=ftmp;
    for (il=1; il < nl-2; il++) {
            fgets (str , 200, pFile);
//            printf("%ld: %s\n",il,str);
            sscanf (str,"%lf %lf",&xtmp,&ftmp);
            xpixels[il]=xtmp;
            pixels[il]=ftmp;
            meanx+=xpixels[il]-xpixels[il-1];
    }
    fclose (pFile);
    *crval1=xpixels[0];
    *npoints=nl-2;
    *cdelta1mean=meanx/(nl-2);
    
}


/*
 * Find the first existing column among the names in 'names'.
 * Returns 1 and the column number on success, 0 otherwise.
 */
static int fits_find_column(fitsfile *fptr, const char *names[], int *colnum, const char **found){
    int i, status = 0;
    *colnum = 0;
    *found = NULL;
    for (i = 0; names[i] != NULL; i++) {
        status = 0;
        if (!fits_get_colnum(fptr, CASEINSEN, (char *) names[i], colnum, &status)
                && *colnum > 0) {
            *found = names[i];
            return 1;
        }
        status = 0;
    }
    return 0;
}


/*
 * Convert the wavelength unit stored in TUNIT to a scale factor that
 * converts the column values to Angstroms.
 */
static void fits_wavelength_unit_scale(const char *tunit, double *scale){
    char low[FLEN_VALUE];
    int i;
    *scale = 1.0;
    if (tunit == NULL)
        return;
    for (i = 0; tunit[i] != '\0' && i < FLEN_VALUE - 1; i++)
        low[i] = (char) tolower((unsigned char) tunit[i]);
    low[i] = '\0';

    if (strstr(low, "nanometer") != NULL || strstr(low, "nm") != NULL)
        *scale = 10.0;
    else if (strstr(low, "micron") != NULL || strstr(low, "\xc2\xb5m") != NULL)
        *scale = 10000.0;
    else if (strstr(low, "angstrom") != NULL || strstr(low, "\xc3\x85") != NULL)
        *scale = 1.0;
}


/*
 * Read a spectrum stored as a FITS image (1D or 2D) using the WCS
 * keywords CDELT1, CRVAL1 and (optionally) CRPIX1.
 * Returns 1 on success, 0 if the current HDU is not a suitable image.
 */
static int fits_read_image_hdu(fitsfile *fptr, long *npoints, double **pixels2, double **xpixels2, double *cdelta1, double *crval1){
    int status = 0;
    int bitpix = 0, naxis = 0;
    long naxes[2] = {1, 1};
    long fpixel[2] = {1, 1};
    long ii;
    double caga;
    double crpix1 = 1.0;
    double *pixels, *xpixels;

    if (fits_get_img_param(fptr, 2, &bitpix, &naxis, naxes, &status) || status)
        return 0;
    if (naxis < 1 || naxis > 2)
        return 0;

    /* The WCS keywords must be readable, otherwise this is not a
       standard image spectrum and the caller should try the table path. */
    status = 0;
    if (fits_read_key(fptr, TDOUBLE, "CDELT1", &caga, NULL, &status) || status)
        return 0;
    *cdelta1 = caga;

    status = 0;
    if (fits_read_key(fptr, TDOUBLE, "CRVAL1", &caga, NULL, &status) || status)
        return 0;
    *crval1 = caga;

    status = 0;
    if (fits_read_key(fptr, TDOUBLE, "CRPIX1", &crpix1, NULL, &status))
        crpix1 = 1.0;
    status = 0;

    pixels = (double *) malloc(naxes[0] * sizeof(double));
    xpixels = (double *) malloc(naxes[0] * sizeof(double));
    if (pixels == NULL || xpixels == NULL) {
            printf("Memory allocation error\n");
            exit(1);
    }

    if (*crval1 < 2500){
        printf("Assuming that input units are nanometers instead of Angstroms...\n");
        *crval1 = *crval1 * 10.;
        *cdelta1 = *cdelta1 * 10.;
    }

    /* Read the first row of the image (the previous implementation
       ended up with this same row). */
    fpixel[0] = 1;
    fpixel[1] = 1;
    if (fits_read_pix(fptr, TDOUBLE, fpixel, naxes[0], NULL, pixels, NULL, &status) || status) {
        free(pixels);
        free(xpixels);
        return 0;
    }

    for (ii = 0; ii < naxes[0]; ii++)
        xpixels[ii] = *crval1 + (ii + 1.0 - crpix1) * (*cdelta1);

    *npoints = naxes[0];
    *pixels2 = pixels;
    *xpixels2 = xpixels;
    return 1;
}


/*
 * Read a spectrum stored in a FITS table.
 *
 * Supported layouts:
 *   - scalar columns (one row per pixel): repeat == 1, npoints = nrows
 *   - vector columns with a single row: nrows == 1, npoints = repeat
 * Tables with several rows of vector columns are rejected.
 *
 * Returns 1 on success, 0 otherwise.
 */
static int fits_read_table_hdu(fitsfile *fptr, int hdu, long *npoints, double **pixels2, double **xpixels2, double *cdelta1, double *crval1){
    const char *wave_names[] = {"WAVE_AIR", "WAVE", "WAVELENGTH", NULL};
    const char *flux_names[] = {"FLUX_EL", "FLUX", "FLUX_CAL", NULL};
    int status = 0;
    int wcol = 0, fcol = 0;
    const char *wcolname = NULL, *fcolname = NULL;
    int typecode = 0, anynul = 0;
    long nrows = 0, repeat_w = 0, repeat_f = 0, width = 0;
    long nw = 0, nf = 0, ii;
    double *wavel = NULL, *flux = NULL;
    double nulval = 0.0;
    double sumdw = 0.0;
    double wscale = 1.0;
    int has_unit = 0;
    long nonmono = 0;
    char tunit[FLEN_VALUE] = "";
    char colkey[FLEN_KEYWORD];

    if (fits_get_num_rows(fptr, &nrows, &status) || status || nrows < 1) {
        status = 0;
        return 0;
    }

    if (!fits_find_column(fptr, wave_names, &wcol, &wcolname))
        return 0;
    if (!fits_find_column(fptr, flux_names, &fcol, &fcolname))
        return 0;

    status = 0;
    if (fits_get_coltype(fptr, wcol, &typecode, &repeat_w, &width, &status) || status) {
        status = 0;
        return 0;
    }
    status = 0;
    if (fits_get_coltype(fptr, fcol, &typecode, &repeat_f, &width, &status) || status) {
        status = 0;
        return 0;
    }

    /* Number of elements of the wavelength column. */
    if (repeat_w == 1)
        nw = nrows;
    else if (nrows == 1)
        nw = repeat_w;
    else {
        printf("Error: FITS table in HDU %d has %ld rows of vector columns (repeat = %ld).\n",
               hdu, nrows, repeat_w);
        printf("Only one row of vector columns or scalar columns (one row per pixel) are supported.\n");
        return 0;
    }

    /* Number of elements of the flux column. */
    if (repeat_f == 1)
        nf = nrows;
    else if (nrows == 1)
        nf = repeat_f;
    else {
        printf("Error: FITS table in HDU %d has %ld rows of vector columns (repeat = %ld).\n",
               hdu, nrows, repeat_f);
        printf("Only one row of vector columns or scalar columns (one row per pixel) are supported.\n");
        return 0;
    }

    if (nw != nf || nw < 2) {
        printf("Error: incompatible wavelength/flux columns in FITS table HDU %d (%ld vs %ld elements).\n",
               hdu, nw, nf);
        return 0;
    }

    wavel = (double *) malloc(nw * sizeof(double));
    flux = (double *) malloc(nw * sizeof(double));
    if (wavel == NULL || flux == NULL) {
        printf("Memory allocation error\n");
        exit(1);
    }

    status = 0;
    if (fits_read_col(fptr, TDOUBLE, wcol, 1, 1, nw, &nulval, wavel, &anynul, &status) || status) {
        printf("Error: unable to read wavelength column from FITS table HDU %d.\n", hdu);
        status = 0;
        free(wavel);
        free(flux);
        return 0;
    }

    status = 0;
    if (fits_read_col(fptr, TDOUBLE, fcol, 1, 1, nw, &nulval, flux, &anynul, &status) || status) {
        printf("Error: unable to read flux column from FITS table HDU %d.\n", hdu);
        status = 0;
        free(wavel);
        free(flux);
        return 0;
    }

    /* Wavelength units: use TUNITn when available, otherwise fall back
       to the historical heuristic (values below 2500 are probably nm). */
    sprintf(colkey, "TUNIT%d", wcol);
    status = 0;
    if (!fits_read_key(fptr, TSTRING, colkey, tunit, NULL, &status)) {
        for (ii = 0; tunit[ii] != '\0'; ii++) {
            if (isalnum((unsigned char) tunit[ii])) {
                has_unit = 1;
                break;
            }
        }
    }
    status = 0;

    if (has_unit) {
        fits_wavelength_unit_scale(tunit, &wscale);
        if (wscale != 1.0)
            for (ii = 0; ii < nw; ii++)
                wavel[ii] *= wscale;
    } else if (wavel[0] < 2500.) {
        printf("Assuming that input units are nanometers instead of Angstroms...\n");
        for (ii = 0; ii < nw; ii++)
            wavel[ii] *= 10.;
    }

    for (ii = 1; ii < nw; ii++) {
        if (wavel[ii] <= wavel[ii-1])
            nonmono++;
        sumdw += wavel[ii] - wavel[ii-1];
    }
    if (nonmono)
        printf("Warning: wavelength column is not strictly increasing (%ld non-increasing steps).\n", nonmono);

    printf("FITS table found in HDU %d: wavelength column '%s', flux column '%s'\n",
           hdu, wcolname != NULL ? wcolname : "?", fcolname != NULL ? fcolname : "?");

    *npoints = nw;
    *xpixels2 = wavel;
    *pixels2 = flux;
    *crval1 = wavel[0];
    *cdelta1 = sumdw / (nw - 1);
    printf("FITS table spectrum: %ld points, wavelength range [%.4f, %.4f] Angstrom, mean step %.6g Angstrom\n",
           *npoints, wavel[0], wavel[nw-1], *cdelta1);
    return 1;
}


void read_fits_file(char * filetest, long* npoints, double** pixels2, double** xpixels2, double* cdelta1, double* crval1){
    printf("READING FITS FILE: %s \n", filetest);
    int status = 0;
    int ok = 0;
    int hdu, nhdus = 0, hdutype = 0;
    fitsfile *fptr = NULL;

    if (fits_open_file(&fptr, filetest, READONLY, &status) || status) {
        fits_report_error(stderr, status);
        fflush(stdout);
        fprintf(stderr, "Error: unable to open FITS file '%s'.\n", filetest);
        exit(1);
    }
    printf("Ficheiro Aberto...\n");

    /* First, the classical case: a 1D/2D image with WCS keywords in the
       current (primary) HDU. */
    ok = fits_read_image_hdu(fptr, npoints, pixels2, xpixels2, cdelta1, crval1);

    /* If that failed, scan every HDU looking for a table with
       wavelength/flux columns, or for an image spectrum in an
       extension. */
    if (!ok) {
        status = 0;
        if (!fits_get_num_hdus(fptr, &nhdus, &status) && !status) {
            for (hdu = 1; hdu <= nhdus && !ok; hdu++) {
                status = 0;
                if (fits_movabs_hdu(fptr, hdu, &hdutype, &status) || status) {
                    status = 0;
                    continue;
                }
                if (hdutype == BINARY_TBL || hdutype == ASCII_TBL) {
                    ok = fits_read_table_hdu(fptr, hdu, npoints, pixels2, xpixels2, cdelta1, crval1);
                } else if (hdutype == IMAGE_HDU) {
                    ok = fits_read_image_hdu(fptr, npoints, pixels2, xpixels2, cdelta1, crval1);
                }
                status = 0;
            }
        }
        status = 0;
    }

    fits_close_file(fptr, &status);

    if (!ok) {
        fflush(stdout);
        fprintf(stderr, "Error: no spectrum found in FITS file '%s'.\n", filetest);
        fprintf(stderr, "Supported formats: 1D/2D image with CDELT1/CRVAL1 WCS, or a table with\n");
        fprintf(stderr, "wavelength (WAVE_AIR/WAVE/WAVELENGTH) and flux (FLUX_EL/FLUX/FLUX_CAL) columns.\n");
        exit(1);
    }
}



void read_mine(char * mine, char* filetest, char* fileleitura, char* fileout, double* lambdai, double* lambdaf, double* smoothder, double* space, char* tree, double* distlinha, double* miniline, int* plots_flag, char* rvmask) {

    
    long nl;
    
    nl = file_lines(mine)-1;
    printf("number of options: %ld\n", nl);
    
    FILE * pFile3;
    pFile3= fopen ("logARES.txt","wt");
    fprintf(pFile3,"Input File ARES...\n");


    
    FILE * fopt = fopen(mine,"rt");
    char str[200];
    system("clear");
    printf("Input Parameters:\n\n");
    fprintf(pFile3,"Input Parameters:\n\n");

    fgets (str , 200 , fopt);
    char *pch;
    pch = strtok (str,"' ");
    pch = strtok (NULL, "' ");
    strcpy (filetest,pch);
    printf("specfits: %s\n",filetest);
    fprintf(pFile3,"specfits: %s\n",filetest);


    fgets (str , 200 , fopt);
    pch = strtok (str,"' ");
    pch = strtok (NULL, "' ");
    strcpy (fileleitura,pch);
    printf("readlinedat: %s\n",fileleitura);
    fprintf(pFile3,"readlinedat: %s\n",fileleitura);


    fgets (str , 200 , fopt);
    pch = strtok (str,"' ");
    pch = strtok (NULL, "' ");
    strcpy (fileout,pch);
    printf("fileout: %s\n",fileout);
    fprintf(pFile3,"fileout: %s\n",fileout);


    fgets (str , 200 , fopt);
    pch = strtok (str,"=");
    pch = strtok (NULL, " ");
    *lambdai=atof(pch);
    printf("lambdai: %6.1f\n",*lambdai);
    fprintf(pFile3,"lambdai: %6.1f\n",*lambdai);


    fgets (str , 200 , fopt);
    pch = strtok (str,"=");
    pch = strtok (NULL, " ");
    *lambdaf=atof(pch);
    printf("lambdaf: %6.1f\n",*lambdaf);
    fprintf(pFile3,"lambdaf: %6.1f\n",*lambdaf);


    fgets (str , 200 , fopt);
    pch = strtok (str,"=");
    pch = strtok (NULL, " ");
    *smoothder=atof(pch);
    printf("smoothder: %3.1f\n",*smoothder);
    fprintf(pFile3,"smoothder: %3.1f\n",*smoothder);


    fgets (str , 200 , fopt);
    pch = strtok (str,"=");
    pch = strtok (NULL, " ");
    *space=atol(pch);
    printf("space: %.2f \n",(float) *space);
    fprintf(pFile3,"space: %.2f \n",(float) *space);


//    fgets (str , 200 , fopt);
//    pch = strtok (str,"=");
//    pch = strtok (NULL, " ");
//    *tree=atof(pch);
//    printf("tree: %5.3f\n",*tree);
//    fprintf(pFile3,"tree: %5.3f\n",*tree);
    fgets (str , 200 , fopt);
    pch = strtok (str,"=");
    pch = strtok (NULL, " ");
    strcpy (tree,pch);
//    printf("tree: %s\n",tree);
//    fprintf(pFile3,"tree: %s\n",tree);
    printf("rejt: %s\n",tree);
    fprintf(pFile3,"rejt: %s\n",tree);


    fgets (str , 200 , fopt);
    pch = strtok (str,"=");
    pch = strtok (NULL, " ");
    *distlinha=atof(pch);
    printf("lineresol: %4.3f\n",*distlinha);
    fprintf(pFile3,"lineresol: %4.3f\n",*distlinha);


    fgets (str , 200 , fopt);
    pch = strtok (str,"=");
    pch = strtok (NULL, " ");
    *miniline=atof(pch);
    printf("miniline: %3.1f\n",*miniline);
    fprintf(pFile3,"miniline: %3.1f\n",*miniline);


    fgets (str , 200 , fopt);
    pch = strtok (str,"=");
    pch = strtok (NULL, " ");
    *plots_flag=atoi(pch);
    printf("plots_flag: %i\n",*plots_flag);
    fprintf(pFile3,"plots_flag: %i\n",*plots_flag);
    
    if (nl > 11){
        fgets (str , 200 , fopt);
        pch = strtok (str,"' ");
        pch = strtok (NULL, "' ");
        strcpy (rvmask,pch);
        printf("rvmask: %s\n",rvmask);
        fprintf(pFile3,"rvmask: %s\n",rvmask);
    } else {
        strcpy (rvmask,"0,0");
    }


    fclose (fopt);
    fclose (pFile3);


}

#ifdef	__cplusplus
}
#endif

#endif	/* _FILESIO_H */

