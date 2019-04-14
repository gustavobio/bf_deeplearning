from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

model_file_url = 'https://drive.google.com/file/d/1d2DqiOSG3JX2T8Ij9GtOCg8rMonu7MnR/view?usp=sharing'
model_file_name = 'model'
classes = ['Amazilia_brevirostris', 'Amazilia_chionogaster', 'Amazilia_fimbriata', 'Amazilia_lactea', 'Amazilia_leucogaster', 'Amazilia_rondoniae', 'Amazilia_sp.', 'Amazilia_versicolor', 'Amazilia_viridigaster', 'Anthracothorax_nigricollis', 'Anthracothorax_viridigula', 'Aphantochroa_cirrochloris', 'Augastes_lumachella', 'Augastes_scutatus', 'Avocettula_recurvirostris', 'Calliphlox_amethystina', 'Campylopterus_duidae', 'Campylopterus_hyperythrus', 'Campylopterus_largipennis', 'Chlorestes_notata', 'Chlorostilbon_lucidus', 'Chlorostilbon_mellisugus', 'Chrysolampis_mosquitus', 'Chrysuronia_oenone', 'Colibri_coruscans', 'Colibri_delphinae', 'Colibri_serrirostris', 'Discosura_langsdorffi', 'Discosura_longicaudus', 'Doryfera_johannae', 'Eupetomena_macroura', 'Florisuga_fusca', 'Florisuga_mellivora', 'Heliactin_bilophus', 'Heliodoxa_aurescens', 'Heliodoxa_rubricauda', 'Heliodoxa_schreibersii', 'Heliodoxa_xanthogonys', 'Heliomaster_furcifer', 'Heliomaster_longirostris', 'Heliomaster_sp.', 'Heliomaster_squamosus', 'Heliothryx_auritus', 'Hylocharis_chrysura', 'Hylocharis_cyanus', 'Hylocharis_sapphirina', 'Leucippus_chlorocercus', 'Leucochloris_albicollis', 'Lophornis_chalybeus', 'Lophornis_gouldii', 'Lophornis_magnificus', 'Lophornis_ornatus', 'Lophornis_pavoninus', 'Polytmus_guainumbi', 'Polytmus_theresiae', 'Stephanoxis_lalandi', 'Stephanoxis_loddigesii', 'Thalurania_furcata', 'Thalurania_glaucopis', 'Thalurania_sp.', 'Thalurania_watertonii', 'Topaza_pella', 'Topaza_pyra']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    data_bunch = ImageDataBunch.single_from_classes(path, classes,
        tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = cnn_learner(data_bunch, models.resnet34, pretrained=False)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    return JSONResponse({'result': learn.predict(img)[0]})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)

