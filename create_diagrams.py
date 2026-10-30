"""Generate five original, high-resolution protocol diagrams; never performance plots."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT=Path(__file__).parent/'figures'; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams['font.family']='DejaVu Sans'

def canvas(h=3.4):
    fig,ax=plt.subplots(figsize=(7.2,h));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off');return fig,ax

def box(ax,x,y,w,h,title,sub='',color='#FAFCFD',fs=12.1):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=.013',
                  edgecolor='#2C4258',facecolor=color,lw=1.1,zorder=2))
    ax.text(x+w/2,y+h*(.74 if sub else .5),title,fontsize=fs,weight='semibold',
            ha='center',va='center',color='#152B42',zorder=3)
    if sub:
        ax.text(x+w/2,y+h*.23,sub,fontsize=11.7,ha='center',va='center',
                color='#33475A',zorder=3)

def arr(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',color='#344A60',
                 linewidth=1.4,mutation_scale=14,zorder=1))

def note(ax,text,y=.08):
    ax.text(.5,y,text,fontsize=11.65,ha='center',va='center',color='#8A3030')

def save(fig,name):
    fig.savefig(OUT/name,dpi=260,bbox_inches='tight',pad_inches=.10,facecolor='white')
    plt.close(fig)

fig,ax=canvas(3.25)
for i,(head,sub) in enumerate([('Cohorts','Unpaired'),('Tasks','One modality'),
                              ('Splits','By patient'),('Reports','Separate')]):
    x=.025+.247*i;box(ax,x,.65,.209,.22,head,sub)
    if i<3:arr(ax,(x+.211,.76),(x+.247,.76))
ax.text(.5,.56,'Independent datasets cannot establish real tri-modal fusion.',
        ha='center',va='center',fontsize=12.3,color='#852C31',weight='semibold')
for i,(head,sub) in enumerate([('MIMIC-IV','Linked patients'),('Attention','Valid streams'),('FL clients','Train only')]):
    x=.105+.295*i;box(ax,x,.20,.215,.22,head,sub,color='#EAF3F9')
    if i<2:arr(ax,(x+.215,.31),(x+.295,.31))
note(ax,'Only the matched cohort can support patient-level fusion.',.075)
save(fig,'fig1_cohort_boundary.png')

fig,ax=canvas(3.3)
for x,head,sub in [(.025,'Echo video','MIMIC-IV'),(.365,'ECG signal','MIMIC-IV'),(.705,'Clinical EHR','MIMIC-IV')]:
    box(ax,x,.67,.267,.22,head,sub)
    arr(ax,(x+.1335,.67),(.5,.53))
box(ax,.255,.36,.49,.17,'Patient linkage','Subject ID + event time',color='#E6F1F8')
arr(ax,(.5,.36),(.5,.26))
box(ax,.315,.10,.37,.16,'Eligible cohort','Observed records',color='#EDF5ED')
note(ax,'Exclude unmatched patients and post-index features.',.032)
save(fig,'fig3_patient_linkage.png')

fig,ax=canvas(3.5)
box(ax,.33,.77,.34,.14,'Linked cohort','One index / person',color='#F3F8FA')
mid=[(.025,'TRAIN: 60%'),(.275,'TUNE: 15%'),(.525,'CAL: 10%'),(.775,'TEST: 15%')]
for i,(x,title) in enumerate(mid):
    box(ax,x,.41,.198,.19,title, color=['#EAF4FC','#F8FAFC','#FBF3E8','#ECF8EC'][i],fs=11.8)
    arr(ax,(.5,.77),(x+.099,.60))
box(ax,.025,.14,.198,.13,'5 FL clients',color='#EAF4FC',fs=11.7)
arr(ax,(.124,.41),(.124,.27))
note(ax,'TEST remains untouched; virtual clients are not hospitals.',.065)
save(fig,'fig4_locked_evaluation.png')

fig,ax=canvas(3.5)
for x,y,title in [(.02,.69,'Echo tokens'),(.02,.415,'ECG tokens'),(.02,.14,'EHR tokens')]:
    box(ax,x,y,.235,.19,title,'Quality + time')
    arr(ax,(x+.235,y+.095),(.39,.505))
box(ax,.39,.355,.31,.30,'Masked attention','Presence + lag',color='#E6F1FA')
box(ax,.785,.62,.195,.19,'Risk score','Calibrate',color='#EBF7EB',fs=11.8)
box(ax,.785,.23,.195,.19,'Abstain','No evidence',color='#FFF3E9',fs=11.8)
arr(ax,(.70,.55),(.785,.715));arr(ax,(.70,.46),(.785,.325))
note(ax,'Unavailable modality embeddings must not affect risk.',.055)
save(fig,'fig5_masked_attention.png')

fig,ax=canvas(2.85)
for i,(title,sub) in enumerate([('Register','Endpoint'),('Freeze','Patient split'),
                                ('Calibrate','Test closed'),('Evaluate','CI + audit')]):
    x=.02+.245*i;box(ax,x,.43,.21,.30,title,sub)
    if i<3:arr(ax,(x+.21,.58),(x+.245,.58))
ax.text(.5,.26,'Report class counts, modality masks, calibration and uncertainty',
        fontsize=11.8,ha='center',color='#31485C')
ax.text(.5,.16,'Track privacy settings, matched baselines and edge measurement logs',
        fontsize=11.8,ha='center',color='#31485C')
note(ax,'Prospective targets are not empirical results.',.06)
save(fig,'fig2_reporting_gates.png')
print('Generated 5 diagrams in',OUT)
