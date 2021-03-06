#!/bin/bash
# set path to watch
run=$1
folder='/run/user/1003/gvfs/smb-share:server=opn.cdb.nas.csc.es,share=opentrons/RUNS/'
#folder="/run/user/1003/gvfs/smb-share:server=cscfs2,share=usr2/USERS/OPENTRONS/RUNS/"

rscript=`find ${folder}${run} -name '*.Rmd' -print0 | xargs -r0 echo | cut -d '/' -f10`
#rscript=`find ${folder}${run} -name '*.Rmd' -print0 | xargs -r0 echo | cut -d '/' -f12` #backup folder
Rscript -e 'library(rmarkdown);
rmarkdown::render("'${folder}${run}'/scripts/'${rscript}'",
"html_document", output_file = "'${run}'_resultados.html",
output_dir="'$folder${run}'/results/")'

echo ${rscript}' executed for run '${run}
