public class ProjectRepositoryImpl implements ProjectRepository{
    private APIService service = ServiceGenerator.createService(APIService.class);
    private static final String TAG = ProjectRepositoryImpl.class.getCanonicalName();

    @Override
    public MutableLiveData<BusinessResult<ProyectoModel>> findAllProyectosByUser(Integer id, String key) {
        MutableLiveData<BusinessResult<ProyectoModel>> proyectoDataMutableLiveData = new MutableLiveData<>();

        try {
            service.getProyectosByUsuario(id, key).enqueue(new Callback<List<ProyectoData>>() {
                @Override
                public void onResponse(Call<List<ProyectoData>> call, Response<List<ProyectoData>> response) {
                    BusinessResult<ProyectoModel> model = new BusinessResult<>();
                    List<ProyectoModel> modelos = new ArrayList<>();
                    if (response.isSuccessful()) {
                        for (ProyectoData data : response.body()) {
                            ProyectoModel proyectoModel = new ProyectoModel();
                            proyectoModel.setRate(data.getCalificacion());
                            proyectoModel.setId(data.getId());
                            proyectoModel.setName(data.getNombre());
                            proyectoModel.setTextDate(data.getFecha());
                            modelos.add(proyectoModel);
                        }
                        model.setResults(modelos);
                        model.setCode(ResultCodes.SUCCESS);
                    }
                    proyectoDataMutableLiveData.setValue(model);
                }
                @Override
                public void onFailure(Call<List<ProyectoData>> call, Throwable t) {
                    BusinessResult<ProyectoModel> model = new BusinessResult<>();
                    proyectoDataMutableLiveData.setValue(model);
                }
            });
        } catch (NetworkOnMainThreadException e) {
            Log.e(TAG, "findAllProyectosByUser ", e);
        }
        return proyectoDataMutableLiveData;
    }
}