# echo "cleanup Luminosity version-1 in /junofs/users/wxfang/FastSim/bes3/workarea"

if ( $?CMTROOT == 0 ) then
  setenv CMTROOT /afs/ihep.ac.cn/bes3/offline/ExternalLib/SLC6/contrib/CMT/v1r25
endif
source ${CMTROOT}/mgr/setup.csh
set cmtLuminositytempfile=`${CMTROOT}/mgr/cmt -quiet build temporary_name`
if $status != 0 then
  set cmtLuminositytempfile=/tmp/cmt.$$
endif
${CMTROOT}/mgr/cmt cleanup -csh -pack=Luminosity -version=version-1 -path=/junofs/users/wxfang/FastSim/bes3/workarea  $* >${cmtLuminositytempfile}
if ( $status != 0 ) then
  echo "${CMTROOT}/mgr/cmt cleanup -csh -pack=Luminosity -version=version-1 -path=/junofs/users/wxfang/FastSim/bes3/workarea  $* >${cmtLuminositytempfile}"
  set cmtcleanupstatus=2
  /bin/rm -f ${cmtLuminositytempfile}
  unset cmtLuminositytempfile
  exit $cmtcleanupstatus
endif
set cmtcleanupstatus=0
source ${cmtLuminositytempfile}
if ( $status != 0 ) then
  set cmtcleanupstatus=2
endif
/bin/rm -f ${cmtLuminositytempfile}
unset cmtLuminositytempfile
exit $cmtcleanupstatus

