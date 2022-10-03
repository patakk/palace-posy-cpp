#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	blurShader.load("shaders/blur");
	postShader.load("shaders/post");

	pixs.allocate(DIMXS, DIMYS, OF_IMAGE_COLOR);
	layer.allocate(DIMXS, DIMYS);
	baseLayer.allocate(DIMXS, DIMYS);
	tempLayer.allocate(DIMXS, DIMYS);

	seed = int(ofRandom(1000000, 9999999));
	ofSeedRandom(seed);
	gui.setup();
	gui.setSize(200, 100);
	gui.add(gvariant.setup("variant", int(ofRandom(3)), 0, 2));
	gui.add(gseed.setup("seed", seed, 0, 99999999));
	gui.add(gscale.setup("scale", sscale, 1, 6));

	variant = gvariant;

	reset();
}

ofFloatColor ofApp::getRandomColor(float p) {

	int start = 0;
	for (int k = 0; k < palettei + 0; k++) {
		start += colori[k];
	}
	int end = start + colori[palettei];
	
	int index = start + int(ofRandom(1) * (end - start));

	float * cc = ofApp::colors[index];

	return ofFloatColor(cc[0], cc[1], cc[2]);
}

void ofApp::reset() {


	int size = *(&colori + 1) - colori;
	ofApp::palettei = int(ofRandom(size));
	cout << palettei << endl;

	//variant = int(ofRandom(3.));
	cout << "variant " << variant << endl;
	cout << "seed    " << seed << endl;
	cout << "scale   " << sscale << endl;

	baseLayer.begin();
	ofBackground(ofRandom(0, 22), ofRandom(0, 22), ofRandom(0, 22));
	ofSetColor(200, 20, 30);
	for (int k = 0; k < 20; k++) {
		ofFloatColor color = getRandomColor(ofRandom(1.));
		ofSetColor(color);
		ofDrawRectangle(ofRandom(0, DIMXS), ofRandom(0, DIMYS), DIMXS *(.3 + .3* ofRandomf()), DIMYS * (.3 + .3 * ofRandomf()));
	}
	baseLayer.end();

	ofBackground(100, 20, 30);
	ofSetColor(255);

	tempLayer.begin();
	baseLayer.draw(0, 0, DIMXS, DIMYS);
	tempLayer.end();

	float blurSeed = ofRandom(1.);

	for (int k = 0; k < 9; k++) {
		layer.begin();
		blurShader.begin();
		blurShader.setUniformTexture("tDiffuse", tempLayer.getTexture(), 0);
		blurShader.setUniform2f("resolution", DIMXS, DIMYS);
		blurShader.setUniform1f("amp", 2);
		blurShader.setUniform1f("variant", variant);
		blurShader.setUniform1f("seed", blurSeed);
		blurShader.setUniform2f("uDir", 1, 0);
		ofDrawRectangle(0, 0, DIMXS, DIMYS);
		blurShader.end();
		layer.end();

		tempLayer.begin();
		layer.draw(0, 0, DIMXS, DIMYS);
		tempLayer.end();

		layer.begin();
		blurShader.begin();
		blurShader.setUniformTexture("tDiffuse", tempLayer.getTexture(), 0);
		blurShader.setUniform2f("resolution", DIMXS, DIMYS);
		blurShader.setUniform1f("amp", .02);
		blurShader.setUniform1f("variant", variant);
		blurShader.setUniform1f("seed", blurSeed);
		blurShader.setUniform2f("uDir", 0, 1);
		ofDrawRectangle(0, 0, DIMXS, DIMYS);
		blurShader.end();
		layer.end();

		tempLayer.begin();
		layer.draw(0, 0, DIMXS, DIMYS);
		tempLayer.end();
	}


	layer.begin();
	postShader.begin();
	postShader.setUniformTexture("tDiffuse4", tempLayer.getTexture(), 0);
	postShader.setUniform2f("resolution", DIMXS, DIMYS);
	postShader.setUniform1f("ztime", DIMXS);
	postShader.setUniform1f("flip", ofRandomf());
	postShader.setUniform1f("seed1", ofRandomf());
	postShader.setUniform1f("seed2", ofRandomf());
	postShader.setUniform1f("seed3", ofRandomf());
	ofDrawRectangle(0, 0, DIMXS, DIMYS);
	postShader.end();
	layer.end();
}

//--------------------------------------------------------------
void ofApp::update() {
	//
	if (gvariant != variant) {
		ofSeedRandom(seed);
		variant = int(ofRandom(3.));
		variant = gvariant;
		reset();
	}
	if (gscale != sscale) {
		sscale = gscale;
		DIMXS = DIMX * sscale;
		DIMYS = DIMY * sscale;

		pixs.allocate(DIMXS, DIMYS, OF_IMAGE_COLOR);
		layer.allocate(DIMXS, DIMYS);
		baseLayer.allocate(DIMXS, DIMYS);
		tempLayer.allocate(DIMXS, DIMYS);

		ofSeedRandom(seed);
		variant = int(ofRandom(3.));
		gvariant = variant;
		reset();
	}
	if (gseed != seed) {
		seed = gseed;
		ofSeedRandom(seed);
		variant = int(ofRandom(3.));
		gvariant = variant;
		reset();
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	if (flag == 0) {
		layer.draw(0, 0, ofGetWidth(), ofGetHeight());
	}
	else {
		baseLayer.draw(0, 0, ofGetWidth(), ofGetHeight());
	}

	gui.draw();
}

//--------------------------------------------------------------

void ofApp::keyPressed(int key) {
	switch (key) {
		case 'r':
			seed = int(ofRandom(1000000, 9999999));
			gseed = seed;
			ofSeedRandom(seed);
			variant = int(ofRandom(3.));
			gvariant = variant;
			reset();
			break;
		case 'f':
			ofToggleFullscreen();
			break;
		case 'w':
			if (flag == 0) {
				flag = 1;
			}
			else {
				flag = 0;
			}
			break;
		case 's':
			ofLogToConsole();
			cout << "saving...\n";
			layer.readToPixels(pixs, 0);
			ofSaveImage(pixs, "untitled_" + ofToString(seed) + "_" + ofToString(DIMXS) + "x" + ofToString(DIMYS) + ".png");
			cout << "saved!\n";
			break;
		default:
			break;
	}
}
//--------------------------------------------------------------
void ofApp::keyReleased(int key) {

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}