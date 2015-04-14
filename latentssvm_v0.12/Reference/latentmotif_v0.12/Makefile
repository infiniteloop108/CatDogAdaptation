# Makefile for Latent Structural SVM

CC=gcc
#CFLAGS= -g -Wall
CFLAGS= -O3 -fomit-frame-pointer -ffast-math
#CFLAGS = -O3 -pg
LD=gcc
#LDFLAGS= -g
LDFLAGS= -O3
#LDFLAGS = -O3 -pg
LIBS= -lm
SFMTPATH= ./SFMT-src-1.3.3

all: svm_motif_learn svm_motif_classify

clean: 
	rm -f *.o
	rm -f svm_latent_learn svm_latent_classify

svm_light_hideo_noexe: 
	cd svm_light; make svm_learn_hideo_noexe

svm_motif_learn: svm_light_hideo_noexe svm_struct_latent_cccp.o svm_struct_latent_api.o SFMT.o
	$(LD) $(LDFLAGS) svm_struct_latent_cccp.o svm_light/svm_hideo.o svm_light/svm_learn.o svm_light/svm_common.o SFMT.o svm_struct_latent_api.o -o svm_motif_learn -lm

svm_motif_classify: svm_light_hideo_noexe svm_struct_latent_classify.o svm_struct_latent_api.o SFMT.o
	$(LD) $(LDFLAGS) svm_struct_latent_classify.o svm_light/svm_common.o SFMT.o svm_struct_latent_api.o -o svm_motif_classify $(LIBS)

svm_struct_latent_cccp.o: svm_struct_latent_cccp.c
	$(CC) -c $(CFLAGS) svm_struct_latent_cccp.c -o svm_struct_latent_cccp.o

svm_struct_latent_api.o: svm_struct_latent_api.c svm_struct_latent_api_types.h
	$(CC) -c $(CFLAGS) svm_struct_latent_api.c -o svm_struct_latent_api.o

svm_struct_latent_classify.o: svm_struct_latent_classify.c
	$(CC) -c $(CFLAGS) svm_struct_latent_classify.c -o svm_struct_latent_classify.o

SFMT.o: 
	$(CC) -c -DMEXP=607 $(SFMTPATH)/SFMT.c -o SFMT.o
