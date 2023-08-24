# echo "cleanup Luminosity version-1 in /junofs/users/wxfang/FastSim/bes3/workarea"

if test "${CMTROOT}" = ""; then
  CMTROOT=/afs/ihep.ac.cn/bes3/offline/ExternalLib/SLC6/contrib/CMT/v1r25; export CMTROOT
fi
. ${CMTROOT}/mgr/setup.sh
cmtLuminositytempfile=`${CMTROOT}/mgr/cmt -quiet build temporary_name`
if test ! $? = 0 ; then cmtLuminositytempfile=/tmp/cmt.$$; fi
${CMTROOT}/mgr/cmt cleanup -sh -pack=Luminosity -version=version-1 -path=/junofs/users/wxfang/FastSim/bes3/workarea  $* >${cmtLuminositytempfile}
if test $? != 0 ; then
  echo >&2 "${CMTROOT}/mgr/cmt cleanup -sh -pack=Luminosity -version=version-1 -path=/junofs/users/wxfang/FastSim/bes3/workarea  $* >${cmtLuminositytempfile}"
  cmtcleanupstatus=2
  /bin/rm -f ${cmtLuminositytempfile}
  unset cmtLuminositytempfile
  return $cmtcleanupstatus
fi
cmtcleanupstatus=0
. ${cmtLuminositytempfile}
if test $? != 0 ; then
  cmtcleanupstatus=2
fi
/bin/rm -f ${cmtLuminositytempfile}
unset cmtLuminositytempfile
return $cmtcleanupstatus

